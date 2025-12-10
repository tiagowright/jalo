"""Memory management for keyboard layouts.

This module provides data structures and a manager for storing and retrieving
keyboard layouts in a stack and numbered lists.
"""

import dataclasses
from typing import Optional

from layout import KeyboardLayout


@dataclasses.dataclass(slots=True)
class StackItem:
    """Represents a single item in the stack (list 0)."""
    layout: KeyboardLayout
    command: str  # Full user input command
    original_layout: Optional[KeyboardLayout] = None  # Original layout if one was given as argument


@dataclasses.dataclass(slots=True)
class LayoutList:
    """Represents a numbered list of layouts."""
    layouts: list[KeyboardLayout]
    list_number: int
    command: str  # Full user input command
    original_layout: Optional[KeyboardLayout] = None  # Original layout if one was given as argument


class LayoutMemoryManager:
    """Manages all layout memory: stack and numbered lists."""
    
    def __init__(self, max_stack_size: int = 100) -> None:
        self._stack: list[StackItem] = []  # Stack: newest is at index 0
        self._lists: dict[int, LayoutList] = {}  # List number -> LayoutList
        self._next_list_number = 1
        self._max_stack_size = max_stack_size
    
    def push_to_stack(self, layout: KeyboardLayout, command: str, original_layout: Optional[KeyboardLayout] = None) -> None:
        """Push a single layout to the stack (list 0)."""
        item = StackItem(layout=layout, command=command, original_layout=original_layout)


        # this is the version where we insert at the beginning of the stack
        # # # Inserting is O(n) because we rename every layout in the stack
        # # self._stack.insert(0, item)  # Insert at beginning (newest first)
        # self._stack = self._stack[:self._max_stack_size]
        
        # # Update layout names to reflect their position in the stack
        # for i, item in enumerate(self._stack):
        #     item.layout.name = f'{i+1}'

        item.layout.name = f'{len(self._stack)+1}'
        self._stack.append(item)

        while len(self._stack) > self._max_stack_size:
            self._stack.pop(0)
        
        
    
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
    
    def get_stack_item(self, index: int) -> Optional[StackItem]:
        """Get a stack item by index (1-based, where 1 is newest)."""
        if index < 1 or index > len(self._stack):
            return None
        return self._stack[index - 1]
    
    def get_stack_items(self, top_n: int) -> list[StackItem]:
        """Get the top N items from the stack."""
        top_n = min(top_n, len(self._stack))

        return self._stack[-top_n:]
    
    def get_list(self, list_num: int) -> Optional[LayoutList]:
        """Get a numbered list by its number."""
        return self._lists.get(list_num)
    
    def get_layout_from_stack(self, index: int) -> Optional[KeyboardLayout]:
        """Get a layout from the stack by index (1-based)."""
        item = self.get_stack_item(index)
        return item.layout if item else None
    
    def get_layout_from_list(self, list_num: int, layout_index: int) -> Optional[KeyboardLayout]:
        """Get a layout from a numbered list."""
        layout_list = self.get_list(list_num)
        if not layout_list:
            return None
        if layout_index < 1 or layout_index > len(layout_list.layouts):
            return None
        return layout_list.layouts[layout_index - 1]
    
    def get_list_layouts(self, list_num: int, top_n: int) -> Optional[list[KeyboardLayout]]:
        """Get the top N layouts from a numbered list."""
        top_n = min(top_n, len(self._lists[list_num].layouts))
        layout_list = self.get_list(list_num)
        if not layout_list:
            return None
        return layout_list.layouts[:top_n]
    
    def get_list_layout_count(self, list_num: int) -> Optional[int]:
        """Get the number of layouts in a numbered list."""
        layout_list = self.get_list(list_num)
        if not layout_list:
            return None
        return len(layout_list.layouts)
    
    def get_list_command(self, list_num: int) -> Optional[str]:
        """Get the command that created a numbered list."""
        layout_list = self.get_list(list_num)
        return layout_list.command if layout_list else None
    
    def get_list_original_layout(self, list_num: int) -> Optional[KeyboardLayout]:
        """Get the original layout for a numbered list."""
        layout_list = self.get_list(list_num)
        return layout_list.original_layout if layout_list else None
    
    def get_all_list_numbers(self) -> list[int]:
        """Get all list numbers in sorted order."""
        return sorted(self._lists.keys())
    
    def stack_size(self) -> int:
        """Get the size of the stack."""
        return len(self._stack)
    
    def has_stack_items(self) -> bool:
        """Check if the stack has any items."""
        return len(self._stack) > 0

