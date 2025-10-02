"""MLP utilities and models for mllf.

This module deliberately avoids importing heavy submodules at package import
time to keep test collection lightweight. Import model/data_split lazily.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
	# for type checkers, expose the symbols
	from .model import SimpleMLP  # noqa: F401
	from .data_split import split_train_val_test  # noqa: F401

__all__ = ["SimpleMLP", "split_train_val_test"]


def load_model_module():
	from .model import SimpleMLP

	return SimpleMLP


def load_data_split():
	from .data_split import split_train_val_test

	return split_train_val_test
