import logging
import random
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import numpy as np

from utils import freeze_model, unfreeze_model, stack_embeddings_to_dataset, to_numpy, smote_augmentation_classwise
from tabpfn.scripts.transformer_prediction_interface import TabPFNEncoder

TABPFN_HIDDEN_DIM = 512

logging.basicConfig(level=logging.INFO,
                    format=f'[{__name__}:%(levelname)s] %(message)s')


class TabMDA(nn.Module):
    def __init__(self, encoder, classifier, freeze_encoder=True, device='cpu'):
        """
        General module which performs (i) encoding and (ii) classification on the encoded data.

        Parameters:
        - encoder: A model that implements .encode() (e.g., TabPFNEncoder)
        - classifier: A model that implements .forward()
        """
        super().__init__()

        if not isinstance(encoder[0], TabPFNEncoder):
            raise ValueError(f'Unsupported encoder type {type(self.encoder_class)}')

        if classifier and not isinstance(classifier, nn.Module):
            raise ValueError(f'Unsupported classifier type {type(classifier)}. '
                             f'It has to be a subclass of torch.nn.Module')

        self.encoder_class = encoder[0]   # This is just the TabPFNEncoder class with the model and some methods. It's not the actual model
        self.encoder_module = encoder[1]  # Extract the actual nn.Module with the TabPFN model so that we can optimize it

        self.classifier = classifier
        
        self.freeze_encoder = freeze_encoder

        if not self.freeze_encoder:
            unfreeze_model(self.encoder_module)
        else:
            freeze_model(self.encoder_module)
            logging.info(f'[Model] Freezing the encoder')

        # === Move the model to the device===
        self.device = device
        self.to(self.device)

    def forward(self, x):
        return self.predict(x)

    def classify(self, x, as_probabilities=False):
        return self.classifier(x, as_probabilities=as_probabilities)

    def predict(self, x, as_probabilities=False):
        raise NotImplementedError("Not implemented. It must take the context as input. Use the `encode` + `classify` methods instead.")

    def encode(self, X_to_encode, X_context, y_context):
        """
        TabPFN expects the following arguments:
        - X_to_encode, X_context, y_context
        """
        return self.encoder_class.encode(X_to_encode=X_to_encode, 
                                         X_context=X_context, 
                                         y_context=y_context, 
                                         model=self.encoder_module)

    def encode_batch(self, batch, context_subsetting_params=None, smote_params=None):
        """
        Returns the encoded batch of size (B x num_contexts, embedding_size)
            and the corresponding labels (B x num_contexts)
        """
        x, y = batch["x"], batch["y"]
        x_context = batch["x_context"][0]
        y_context = batch["y_context"][0]

        # How to treat a query row's own (x_i, y_i) when sampling its context.
        #   "random"  : current behaviour (a row may or may not land in its own context)
        #   "exclude" : a row is NEVER in its own context (strict leave-one-out)
        #   "include" : a row is ALWAYS in its own context (its own label is always seen)
        self_context = None
        if isinstance(context_subsetting_params, dict):
            self_context = context_subsetting_params.get("self_context")

        if context_subsetting_params and context_subsetting_params["num_contexts"] > 1:
            if isinstance(context_subsetting_params, dict):
                if all(key in context_subsetting_params.keys() for key in ["num_contexts", "context_size"]):
                    # ==== TrivialAugment style context subsetting ====
                    if context_subsetting_params["context_size"] == 0:
                        # Sample one context size for each of the contexts
                        context_sizes = [random.uniform(0.5, 0.99) for _ in range(context_subsetting_params["num_contexts"])]
                        x_batch_enc = torch.zeros((x.shape[0],
                                                   context_subsetting_params["num_contexts"],
                                                   TABPFN_HIDDEN_DIM))

                        for context_idx, context_size in enumerate(context_sizes):
                            x_context_enc = self.context_subsetting(X_train=x_context,
                                                                    y_train=y_context,
                                                                    num_contexts=1,
                                                                    context_size=context_size,
                                                                    X_to_encode=x,
                                                                    self_context=self_context)

                            x_batch_enc[:, context_idx] = x_context_enc[:, 0, :]

                        x_batch_enc, y_batch = stack_embeddings_to_dataset(x_batch_enc, batch["y"])
                    else:
                        x_batch_enc = self.context_subsetting(X_train=x_context,
                                                              y_train=y_context,
                                                              num_contexts=context_subsetting_params["num_contexts"],
                                                              context_size=context_subsetting_params["context_size"],
                                                              X_to_encode=x,
                                                              self_context=self_context)
                        x_batch_enc, y_batch = stack_embeddings_to_dataset(x_batch_enc, batch["y"])

                    if smote_params:
                        if all(key in smote_params.keys() for key in ["k", "rounds"]):
                            x_batch_enc, y_batch = smote_augmentation_classwise(x_batch_enc, y_batch,
                                                                                k=smote_params["k"],
                                                                                rounds=smote_params["rounds"])
                        else:
                            raise ValueError(f'Unsupported SMOTE arguments {smote_params}')

                    return x_batch_enc, y_batch
                else:
                    raise ValueError(f'Unsupported context subsetting arguments {context_subsetting_params}')
            else:
                raise ValueError(f'Unsupported context subsetting arguments of type {type(context_subsetting_params)}')

        else:
            x_batch_enc = self.encode(X_to_encode=x, 
                                      X_context=x_context, 
                                      y_context=y_context)

            x_batch_enc, y_batch = stack_embeddings_to_dataset(x_batch_enc, batch["y"])

            if smote_params:
                if all(key in smote_params.keys() for key in ["k", "rounds"]):
                    x_batch_enc, y_batch = smote_augmentation_classwise(x_batch_enc, y_batch,
                                                                        k=smote_params["k"],
                                                                        rounds=smote_params["rounds"])
                else:
                    raise ValueError(f'Unsupported SMOTE arguments {smote_params}')

            return x_batch_enc, y_batch

    def context_subsetting(self,
        X_train,                # (np.ndarray) data to be encoded
        y_train,                # (np.ndarray) labels of the data to be encoded
        X_to_encode,            # data to be encoded: (one row), (list of rows), or (batch of B x num_features)
        num_contexts,           # (int) number of contexts to generate
        context_size = None,    # If int, size of each context. If float, proportion of the data to use as context. If None, use the maximum context size.
        seed = 42,              # If not None, set the seed for reproducibility
        self_context = None,    # "random" (default) / "exclude" / "include": whether a query row may appear in its own context
    ):
        """
        Encode X by separately fitting the encoder on `num_contexts` random contexts of size `context_size` from `X_train`.

        `self_context` controls whether a query row's own (x_i, y_i) is allowed in its context:
          - None / "random": no constraint. One shared context is sampled per context index and
                             reused for every query row (the original behaviour).
          - "exclude": each query row's context is sampled from X_train \\ {x_q} (strict leave-one-out).
          - "include": each query row x_q is forced into its own context (context_size-1 other rows
                       are sampled and x_q is appended).

        Returns:
        - a tensor with shape (num_query_rows, num_contexts, embedding_size) containing `num_contexts`
          embeddings per row, each using a different context
        """
        if context_size is None:
            context_size = len(X_train)
        elif context_size > 1:
            if context_size > len(X_train):
                raise ValueError(f'The context size ({context_size}) can not be bigger than the available '
                                 f'context size ({len(X_train)}).')

            context_size = int(context_size)
        else:  # A proportion of the data
            context_size = int(context_size * len(X_train))

        # === If X is one row, then convert to list
        if (isinstance(X_to_encode, np.ndarray) or isinstance(X_to_encode, torch.Tensor)) and len(X_to_encode.shape) == 1:
            X_to_encode = [X_to_encode]

        if isinstance(y_train, torch.Tensor):
            num_classes = len(torch.unique(y_train))
        else:
            num_classes = len(np.unique(y_train))

        # List of seeds used for sampling each context
        seeds_for_contexts = [seed + i for i in range(num_contexts)]

        # Generate an array of indices from 0 to the length of your dataset
        indices = np.arange(len(X_train))

        # === Unconstrained (original) mode: one shared context per context index ===
        if self_context is None or self_context == "random":
            output = []
            for i, _ in enumerate(range(num_contexts)):
                # === Sample a random context (prompt) (stratified sampling) ===
                if context_size == len(X_train):
                    X_subcontext, y_subcontext = X_train, y_train
                else:
                    indices_train, indices_test, _, _ = train_test_split(
                        indices, indices,
                        train_size=min(context_size, len(X_train) - num_classes),
                        random_state=seeds_for_contexts[i],
                        stratify=to_numpy(y_train)
                    )

                    X_subcontext = X_train[indices_train]
                    y_subcontext = y_train[indices_train]

                # === Encode the data using the random context ===
                X_subcontext_encoded = self.encode(X_context=X_subcontext,
                                                   X_to_encode=X_to_encode,
                                                   y_context=y_subcontext)

                output.append(X_subcontext_encoded)

            output = torch.cat(output, 1)
            return output

        if self_context not in ("exclude", "include"):
            raise ValueError(f'Unknown self_context mode {self_context!r}. '
                             f'Expected "random", "exclude", or "include".')

        # === Per-query modes: the context depends on which row is encoded, so we encode one
        #     query row at a time (under efficient_eval_masking queries never attend to each other). ===
        X_tr = X_train if isinstance(X_train, torch.Tensor) else torch.tensor(to_numpy(X_train), dtype=torch.float32, device=self.device)
        y_tr = y_train if isinstance(y_train, torch.Tensor) else torch.tensor(to_numpy(y_train), dtype=torch.long, device=self.device)

        if isinstance(X_to_encode, list):
            X_q = torch.stack([x if isinstance(x, torch.Tensor) else torch.tensor(to_numpy(x), dtype=torch.float32, device=X_tr.device) for x in X_to_encode])
        else:
            X_q = X_to_encode
        if X_q.ndim == 1:
            X_q = X_q.unsqueeze(0)

        # In the train-augmentation setting X_to_encode == X_train, so query row q corresponds
        # to train index q (the "self" excluded/included is row q).
        n_query = X_q.shape[0]

        output = []
        for q in range(n_query):
            pool = indices[indices != q]            # X_train \ {x_q}
            if len(pool) == 0:
                raise ValueError(f'Cannot build a context for row {q}: the training pool is empty.')
            pool_y = to_numpy(y_tr[pool])
            pool_nclasses = len(np.unique(pool_y))
            # include keeps one slot for x_q itself, the rest comes from the other rows
            target = context_size if self_context == "exclude" else context_size - 1

            encs = []
            for i, _ in enumerate(range(num_contexts)):
                if target >= len(pool):
                    idx = np.arange(len(pool))
                else:
                    idx = train_test_split(
                        np.arange(len(pool)), np.arange(len(pool)),
                        train_size=min(target, len(pool) - pool_nclasses),
                        random_state=seeds_for_contexts[i],
                        stratify=pool_y
                    )[0]

                X_sub = X_tr[pool[idx]]
                y_sub = y_tr[pool[idx]]

                if self_context == "include":
                    X_sub = torch.cat([X_sub, X_tr[q:q + 1]], dim=0)
                    y_sub = torch.cat([y_sub, y_tr[q:q + 1]], dim=0)

                enc = self.encode(X_to_encode=X_q[q:q + 1], X_context=X_sub, y_context=y_sub)
                encs.append(enc)

            output.append(torch.cat(encs, 1))

        return torch.cat(output, 0)
