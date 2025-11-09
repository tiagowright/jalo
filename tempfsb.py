qwerty = '''q w e r t y u i o p
a s d f g h j k l ;
z x c v b n m , . /'''

grid = {}
for row, line in enumerate(qwerty.split('\n')):
    for col, char in enumerate(line.split()):
        grid[row*10 + col] = char

print(grid)


print(grid[0], grid[21], sep='', end=' ')
print(grid[1], grid[22], sep='', end=' ')
print(grid[6], grid[27], sep='', end=' ')
print(grid[7], grid[28], sep='', end=' ')
print(grid[8], grid[29], sep='', end=' ')
print(grid[1], grid[20], sep='', end=' ')
print(grid[2], grid[21], sep='', end=' ')
print(grid[3], grid[22], sep='', end=' ')
print(grid[8], grid[27], sep='', end=' ')
print(grid[9], grid[28], sep='', end=' ')

# //pinky->ring 1u stretches
print(grid[0], grid[11], sep='', end=' ')
print(grid[9], grid[18], sep='', end=' ')
print(grid[10], grid[21], sep='', end=' ')
print(grid[19], grid[28], sep='', end=' ')

# //inner index scissors (no qwerty `ni` because of stagger)
print(grid[2], grid[24], sep='', end=' ')
print(grid[22], grid[4], sep='', end=' ')
print(grid[5], grid[27], sep='')