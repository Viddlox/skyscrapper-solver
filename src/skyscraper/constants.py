from typing import List, Set, Deque, Tuple, TypedDict, TypeAlias, Dict
from enum import Enum
from dataclasses import dataclass, field
from collections import deque

MAX_ELIMINATION_CACHE_SIZE = 5000
MAX_INTERSECTION_CACHE_SIZE = 5000
CELL_POE_PREFILL_THRESHOLD = 0.15
CELL_POE_HIDDEN_CLUE_THRESHOLD = 0.15


class Actions(Enum):
    ASSIGN_ROW_PERMUTATION = 1
    ASSIGN_COL_PERMUTATION = 2
    PROPAGATE_CONSTRAINTS_FROM_RESOLVED_CELL = 3


class QueueItem(TypedDict):
    type: Actions
    index: int


Permutation: TypeAlias = Tuple[int, ...]
PermutationSet: TypeAlias = Set[Permutation]
ClueConstraints: TypeAlias = Tuple[int, ...]
IntersectionKey: TypeAlias = Tuple[Permutation, Permutation, int]
Prefill: TypeAlias = Tuple[int, int, int]
# (decision_type, idx, permutation, state_hash)
EliminationKey: TypeAlias = Tuple[str, int, Permutation, int]
EliminationCache: TypeAlias = Dict[EliminationKey, int]


@dataclass
class DecisionPoint:
    decision_type: str
    index: int
    chosen_permutation: Permutation
    eliminated_permutations: PermutationSet


@dataclass
class GameState:
    row_permutations: List[PermutationSet]
    col_permutations: List[PermutationSet]
    assigned_rows: Set[int]
    assigned_cols: Set[int]
    queue: Deque[QueueItem]


@dataclass
class POEGameState:
    grid: List[Set[int]] = field(default_factory=list)
    fixed_cells: Set[int] = field(default_factory=set)
    queue: Deque = field(default_factory=deque)
