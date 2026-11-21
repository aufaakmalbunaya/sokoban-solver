# Sokoban solver

Entry for the make.fun bounty
[sokoban-solver-01-run-20261121](https://make.fun/bounty/sokoban-solver-01-run-20261121)
("Build the best Sokoban solver, 21 November 2026").

`main.ts` is a Node 22 program (TypeScript, erasable syntax only, no
dependencies). It reads every instance of a set from `/input/<id>/level.txt`
(XSB Sokoban levels) and writes `/output/<id>/solution.txt` with the
player's moves (`u`/`d`/`l`/`r`), in as few pushes as possible, in one run
for the whole set.

## How it works

Every level in this competition is a 1-wide hallway with 1-4 small rooms
(3-4 boxes each) off it. A box pushed out of a room through its door can
never come back, so each room is an independent puzzle and the level's
push-optimal count is the sum of the rooms' optima. The solver therefore:

1. Splits a level into rooms by cutting the hallway row's dead squares out
   of the floor and flood-filling the parts that hold boxes.
2. Solves each room push-optimally with a breadth-first search over pushes
   (the first solved state met is optimal). A state is the boxes' squares
   plus the player's reachable region, named by its lowest square. The
   search prunes dead squares (found by pulling boxes back from the goals)
   and frozen 2x2 blocks of walls and boxes.
3. Stitches the room solutions together with the walks between them (the
   other rooms' boxes stand as walls while one room is solved).

If a room's breadth-first search ever runs out of its fixed expansion
budget, an A* pass with an admissible box-to-goal assignment heuristic
retries it, then a greedy best-first pass at least produces a solving
answer when one exists. All budgets are fixed expansion counts, never the
clock, so two runs write byte-identical output.

## Results

Practice set (kit): **1,000,000 / 1,000,000** — every instance solved in
exactly its push-optimal count (13, 16, 29, 47, 60 pushes), in about 1.4 s
for the 5-instance set. The house baseline plain-astar 1.0.0 scores 200,000
on the same set. Four extra freshly drawn sets (different generator seeds)
also each scored 1,000,000, in about 0.6-1.0 s per set.

## License

MIT — see `LICENSE`.
