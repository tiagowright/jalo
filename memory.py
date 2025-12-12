"""Memory management for keyboard layouts.

This module provides data structures and a manager for storing and retrieving
keyboard layouts in numbered lists and a stack (most recently used layouts).
"""

import dataclasses
from typing import Optional

from layout import KeyboardLayout


@dataclasses.dataclass(slots=True)
class LayoutList:
    """Represents a numbered list of layouts."""
    layouts: list[KeyboardLayout]
    list_number: int
    command: str  # Full user input command
    original_layout: Optional[KeyboardLayout] = None  # Original layout if one was given as argument


@dataclasses.dataclass(slots=True)
class StackItemMetadata:
    """Metadata for a single item in the stack."""
    command: str  # Full user input command
    original_layout: Optional[KeyboardLayout] = None  # Original layout if one was given as argument


class LayoutMemoryManager:
    """Manages all layout memory: stack and numbered lists."""
    
    def __init__(self, max_stack_size: int = 100) -> None:
        self.stack: list[KeyboardLayout] = []  # Stack of most recently used layouts
        self.stack_metadata: list[StackItemMetadata] = []  # Metadata for stack items (parallel to layouts)
        self._lists: dict[int, LayoutList] = {}  # List number -> LayoutList
        self._next_list_number = 1
        self._max_stack_size = max_stack_size
    
    @property
    def lists(self) -> dict[int, LayoutList]:
        """Access the lists dictionary directly."""
        return self._lists
    
    def push_to_stack(self, layout: KeyboardLayout, command: str, original_layout: Optional[KeyboardLayout] = None) -> None:
        """Push a single layout to the stack.
        
        This method contains logic for naming layouts and managing max stack size.
        """
        layout.name = f'{len(self.stack)+1}'
        self.stack.append(layout)
        self.stack_metadata.append(StackItemMetadata(command=command, original_layout=original_layout))

        # Enforce max stack size
        while len(self.stack) > self._max_stack_size:
            self.stack.pop(0)
            self.stack_metadata.pop(0)
    
    def add_list(self, layouts: list[KeyboardLayout], command: str, original_layout: Optional[KeyboardLayout] = None) -> int:
        """Add a new numbered list of layouts.
        
        Args:
            layouts: List of layouts to add
            command: The command that created these layouts
            original_layout: Original layout if one was given as argument
            
        Returns:
            The list number assigned to this list
        """
        list_num = self._next_list_number

        # Update layout names to reflect their position in the list
        for i, layout in enumerate(layouts):
            layout.name = f'{list_num}.{i+1}'

        self._lists[list_num] = LayoutList(
            layouts=layouts,
            list_number=list_num,
            command=command,
            original_layout=original_layout
        )
        self._next_list_number += 1
        return list_num
    
    def get_stack_item_metadata(self, index: int) -> Optional[StackItemMetadata]:
        """Get metadata for a stack item by index (1-based).
        
        This method contains logic for index validation.
        """
        if index < 1 or index > len(self.stack):
            return None
        return self.stack_metadata[index - 1]
    
    def get_all_list_numbers(self) -> list[int]:
        """Get all list numbers in sorted order."""
        return sorted(self._lists.keys())

