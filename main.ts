// Sokoban solver: push-optimal search, room by room.
//
// Idea: every level in this competition is a 1-wide hallway with 1-4 small
// rooms (3-4 boxes each) off it. A box pushed out of a room through its door
// can never come back, so each room is an independent puzzle and the level's
// push-optimal count is the sum of the rooms' optima. This program therefore
// splits a level into rooms, solves each room push-optimally with a
// breadth-first search over pushes (the first solved state met is optimal),
// and stitches the room solutions together with the walks between them.
//
// The search state is the boxes' squares plus the player's reachable region
// (named by its lowest square). It prunes two dead positions no push can
// save: a square from which no push can ever reach a goal (dead square,
// found by pulling boxes back from the goals), and a frozen 2x2 block of
// walls and boxes holding a box off its goal. Everything is budgeted by
// fixed expansion counts, never by the clock, so two runs write identical
// bytes. If a room's breadth-first search ever runs out of budget, an A*
// pass with an admissible assignment heuristic retries it, then a greedy
// best-first pass at least produces a solving answer when one exists.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const IN_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Fixed work budgets (expansion counts, never the clock).
const BFS_STATE_CAP = 500_000;
const ASTAR_STATE_CAP = 500_000;
const GREEDY_STATE_CAP = 500_000;

// Directions in a fixed order: up, down, left, right.
const DX = [0, 0, -1, 1];
const DY = [-1, 1, 0, 0];
const STEP_CH = ['u', 'd', 'l', 'r'];
const PUSH_CH = ['U', 'D', 'L', 'R'];

interface Board {
  w: number;
  h: number;
  wall: Uint8Array; // 1 = wall
  goals: number[]; // sorted cell indices
  boxes: number[]; // sorted cell indices
  player: number; // cell index
}

function parseLevel(text: string): Board {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x]!;
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  goals.sort((a, b) => a - b);
  boxes.sort((a, b) => a - b);
  return { w, h, wall, goals, boxes, player };
}

// Squares a lone box can be pushed from and still reach some goal: start
// from the goals and walk pulls backwards (a box on y can be pushed to x
// when the player can stand on the far side, y - d).
function liveSquares(b: Board): Uint8Array {
  const live = new Uint8Array(b.w * b.h);
  const queue: number[] = [];
  for (const g of b.goals) {
    live[g] = 1;
    queue.push(g);
  }
  for (let qi = 0; qi < queue.length; qi++) {
    const x = queue[qi]!;
    for (let k = 0; k < 4; k++) {
      const d = DY[k]! * b.w + DX[k]!;
      const y = x - d;
      if (y < 0 || y >= live.length) continue;
      if (!b.wall[y] && !b.wall[y - d] && !live[y]) {
        live[y] = 1;
        queue.push(y);
      }
    }
  }
  return live;
}

// Shortest walking distances (ignoring boxes) from every cell to its
// nearest goal, for the assignment heuristic.
function goalDistances(b: Board): Int32Array {
  const dist = new Int32Array(b.w * b.h).fill(-1);
  const queue: number[] = [];
  for (const g of b.goals) {
    dist[g] = 0;
    queue.push(g);
  }
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi]!;
    for (let k = 0; k < 4; k++) {
      const q = p + DY[k]! * b.w + DX[k]!;
      if (q < 0 || q >= dist.length) continue;
      if (!b.wall[q] && dist[q] < 0) {
        dist[q] = dist[p]! + 1;
        queue.push(q);
      }
    }
  }
  return dist;
}

// Cheapest way to match each box to a distinct goal (small: at most 4
// boxes, so try every matching). A lower bound on pushes still needed.
function assignmentBound(boxes: number[], dist: Int32Array, goals: number[]): number {
  const n = boxes.length;
  if (n === 0) return 0;
  // dist from box i to goal j, -1 when unreachable.
  const cost: number[][] = [];
  for (let i = 0; i < n; i++) {
    const row: number[] = [];
    for (let j = 0; j < goals.length; j++) row.push(dist[boxes[i]!]! >= 0 ? dist[boxes[i]!]! : 1e9);
    cost.push(row);
  }
  // Brute-force matchings via recursion over goals in fixed order.
  let best = 1e9;
  const used = new Uint8Array(goals.length);
  const rec = (i: number, total: number): void => {
    if (total >= best) return;
    if (i === n) {
      best = total;
      return;
    }
    for (let j = 0; j < goals.length; j++) {
      if (used[j]) continue;
      used[j] = 1;
      rec(i + 1, total + cost[i]![j]!);
      used[j] = 0;
    }
  };
  rec(0, 0);
  return best;
}

class PushSearch {
  b: Board;
  live: Uint8Array;
  dist: Int32Array;
  goalMask: Uint8Array;
  occupied: Uint8Array;
  seen: Int32Array;
  stamp: number;
  stack: Int32Array;
  reach: Uint8Array; // snapshot of the current state's reachable region

  constructor(b: Board) {
    this.b = b;
    this.live = liveSquares(b);
    this.dist = goalDistances(b);
    this.goalMask = new Uint8Array(b.w * b.h);
    for (const g of b.goals) this.goalMask[g] = 1;
    this.occupied = new Uint8Array(b.w * b.h);
    this.seen = new Int32Array(b.w * b.h);
    this.stamp = 0;
    this.stack = new Int32Array(b.w * b.h);
    this.reach = new Uint8Array(b.w * b.h);
  }

  // Flood-fill the player's reachable region from `from` (boxes block).
  // Returns the region's lowest cell; marks cells with the current stamp.
  regionLow(from: number): number {
    const { b, seen, stack } = this;
    const stamp = ++this.stamp;
    let low = from;
    let n = 0;
    stack[n++] = from;
    seen[from] = stamp;
    while (n > 0) {
      const p = stack[--n]!;
      if (p < low) low = p;
      for (let k = 0; k < 4; k++) {
        const q = p + DY[k]! * b.w + DX[k]!;
        if (q < 0 || q >= seen.length) continue;
        if (!b.wall[q] && !this.occupied[q] && seen[q] !== stamp) {
          seen[q] = stamp;
          stack[n++] = q;
        }
      }
    }
    return low;
  }

  // True when the box just pushed onto `t` is caught in a frozen 2x2
  // block: four cells of walls and boxes, at least one box off its goal.
  frozenAt(t: number): boolean {
    const { b, occupied, goalMask } = this;
    const w = b.w;
    const corners = [t, t - 1, t - w, t - w - 1];
    for (const a of corners) {
      const c1 = a + 1;
      const c2 = a + w;
      const c3 = c2 + 1;
      if (a < 0 || c3 >= occupied.length) continue;
      // The block must not wrap around a row edge.
      if (a % w === w - 1) continue;
      const filled =
        (b.wall[a] || occupied[a]) &&
        (b.wall[c1] || occupied[c1]) &&
        (b.wall[c2] || occupied[c2]) &&
        (b.wall[c3] || occupied[c3]);
      if (!filled) continue;
      if (
        (occupied[a] && !goalMask[a]) ||
        (occupied[c1] && !goalMask[c1]) ||
        (occupied[c2] && !goalMask[c2]) ||
        (occupied[c3] && !goalMask[c3])
      ) {
        return true;
      }
    }
    return false;
  }

  solved(boxes: number[]): boolean {
    for (const x of boxes) if (!this.goalMask[x]) return false;
    return true;
  }

  // State key: region's lowest cell, then the sorted box cells.
  keyOf(boxes: number[], low: number): string {
    return low + ':' + boxes.join(',');
  }

  // Shortest walk from `from` to `to` around walls and the boxes in
  // `occupied`, in lower-case letters. Fixed direction order, so the walk
  // is deterministic.
  walk(from: number, to: number): string {
    if (from === to) return '';
    const { b } = this;
    const prev = new Int32Array(b.w * b.h).fill(-1);
    const how = new Array<string>(b.w * b.h);
    const queue = [from];
    prev[from] = from;
    for (let qi = 0; qi < queue.length; qi++) {
      const p = queue[qi]!;
      if (p === to) break;
      for (let k = 0; k < 4; k++) {
        const q = p + DY[k]! * b.w + DX[k]!;
        if (q < 0 || q >= prev.length) continue;
        if (b.wall[q] || this.occupied[q] || prev[q] !== -1) continue;
        prev[q] = p;
        how[q] = STEP_CH[k]!;
        queue.push(q);
      }
    }
    if (prev[to] === -1) return '';
    let s = '';
    for (let p = to; p !== from; p = prev[p]!) s = how[p]! + s;
    return s;
  }

  // Turn a chain of push states into the full move string: walk to each
  // push square, then push.
  renderPath(
    stateBoxes: number[][],
    statePlayer: number[],
    parent: number[],
    pushFrom: number[],
    pushDir: number[],
    found: number,
  ): string {
    const chain: number[] = [];
    for (let s = found; s > 0; s = parent[s]!) chain.push(s);
    chain.reverse();
    let moves = '';
    let player = this.b.player;
    let boxes = this.b.boxes;
    for (const s of chain) {
      const from = pushFrom[s]!;
      const k = pushDir[s]!;
      const d = DY[k]! * this.b.w + DX[k]!;
      for (const x of boxes) this.occupied[x] = 1;
      moves += this.walk(player, from - d);
      for (const x of boxes) this.occupied[x] = 0;
      moves += PUSH_CH[k]!;
      player = from;
      boxes = stateBoxes[s]!;
    }
    return moves;
  }

  // Expand one state: every legal push, calling `visit(nextBoxes, from, dir)`
  // for each child. The parent's reachable region is snapshotted first, so
  // the child-region flood fills inside `visit` cannot disturb it.
  // Returns false when the caller should stop.
  forEachPush(
    boxes: number[],
    player: number,
    visit: (next: number[], from: number, dir: number) => boolean,
  ): void {
    const { b } = this;
    for (const x of boxes) this.occupied[x] = 1;
    this.reach.fill(0);
    this.regionLow(player);
    const stamp = this.stamp;
    for (let i = 0; i < this.reach.length; i++) if (this.seen[i] === stamp) this.reach[i] = 1;
    for (let bi = 0; bi < boxes.length; bi++) {
      const box = boxes[bi]!;
      for (let k = 0; k < 4; k++) {
        const d = DY[k]! * b.w + DX[k]!;
        const stand = box - d;
        const target = box + d;
        if (target < 0 || target >= b.w * b.h) continue;
        if (stand < 0 || stand >= b.w * b.h || !this.reach[stand]) continue; // player cannot stand there
        if (b.wall[target] || this.occupied[target]) continue;
        if (!this.live[target]) continue;
        this.occupied[box] = 0;
        this.occupied[target] = 1;
        const frozen = this.frozenAt(target);
        let next: number[] | null = null;
        if (!frozen) {
          next = boxes.slice();
          next[bi] = target;
          next.sort((x, y) => x - y);
        }
        this.occupied[target] = 0;
        this.occupied[box] = 1;
        if (frozen || next === null) continue;
        if (!visit(next, box, k)) {
          for (const x of boxes) this.occupied[x] = 0;
          return;
        }
      }
    }
    for (const x of boxes) this.occupied[x] = 0;
  }

  // Breadth-first search over pushes. First solved state met is optimal.
  bfs(cap: number): string | null {
    const { b } = this;
    for (const x of b.boxes) if (!this.live[x]) return null;
    if (this.solved(b.boxes)) return '';
    const stateBoxes: number[][] = [b.boxes.slice()];
    const statePlayer: number[] = [b.player];
    const parent: number[] = [-1];
    const pushFrom: number[] = [-1];
    const pushDir: number[] = [-1];
    const seenKeys = new Map<string, number>();
    for (const x of b.boxes) this.occupied[x] = 1;
    const low0 = this.regionLow(b.player);
    for (const x of b.boxes) this.occupied[x] = 0;
    seenKeys.set(this.keyOf(b.boxes, low0), 0);
    let found = -1;
    for (let head = 0; found < 0 && head < stateBoxes.length; head++) {
      const boxes = stateBoxes[head]!;
      let stop = false;
      this.forEachPush(boxes, statePlayer[head]!, (next, from, dir) => {
        // Child region: player stands on `from` after the push.
        const low = this.regionLow(from);
        const key = this.keyOf(next, low);
        if (seenKeys.has(key)) return true;
        if (stateBoxes.length >= cap) {
          stop = true;
          return false;
        }
        const idx = stateBoxes.length;
        seenKeys.set(key, idx);
        stateBoxes.push(next);
        statePlayer.push(from);
        parent.push(head);
        pushFrom.push(from);
        pushDir.push(dir);
        if (this.solved(next)) {
          found = idx;
          return false;
        }
        return true;
      });
      if (stop) return null;
      for (const x of boxes) this.occupied[x] = 0;
    }
    if (found < 0) return null;
    return this.renderPath(stateBoxes, statePlayer, parent, pushFrom, pushDir, found);
  }

  // A* over pushes with the assignment bound as heuristic (admissible: one
  // push moves one box one square, so the bound drops by at most one).
  // Still optimal when it completes; explores far fewer states than BFS.
  astar(cap: number): string | null {
    const { b } = this;
    for (const x of b.boxes) if (!this.live[x]) return null;
    if (this.solved(b.boxes)) return '';
    const stateBoxes: number[][] = [b.boxes.slice()];
    const statePlayer: number[] = [b.player];
    const stateG: number[] = [0];
    const parent: number[] = [-1];
    const pushFrom: number[] = [-1];
    const pushDir: number[] = [-1];
    const seenKeys = new Map<string, number>();
    for (const x of b.boxes) this.occupied[x] = 1;
    const low0 = this.regionLow(b.player);
    for (const x of b.boxes) this.occupied[x] = 0;
    seenKeys.set(this.keyOf(b.boxes, low0), 0);
    // Binary heap of [f, g, state], ties by lower g then lower state index.
    const heap: [number, number, number][] = [];
    const less = (a: [number, number, number], c: [number, number, number]): boolean =>
      a[0] < c[0] || (a[0] === c[0] && (a[1] < c[1] || (a[1] === c[1] && a[2] < c[2])));
    const heapPush = (e: [number, number, number]): void => {
      let i = heap.length;
      heap.push(e);
      while (i > 0) {
        const p = (i - 1) >> 1;
        if (!less(heap[i]!, heap[p]!)) break;
        const t = heap[i]!;
        heap[i] = heap[p]!;
        heap[p] = t;
        i = p;
      }
    };
    const heapPop = (): [number, number, number] => {
      const top = heap[0]!;
      const last = heap.pop()!;
      if (heap.length > 0) {
        heap[0] = last;
        for (let i = 0; ; ) {
          let c = 2 * i + 1;
          if (c >= heap.length) break;
          if (c + 1 < heap.length && less(heap[c + 1]!, heap[c]!)) c++;
          if (!less(heap[c]!, heap[i]!)) break;
          const t = heap[i]!;
          heap[i] = heap[c]!;
          heap[c] = t;
          i = c;
        }
      }
      return top;
    };
    heapPush([assignmentBound(b.boxes, this.dist, b.goals), 0, 0]);
    while (heap.length > 0) {
      const [, g, s] = heapPop();
      if (seenKeys.get(this.keyOf(stateBoxes[s]!, this.regionLowForKey(stateBoxes[s]!, statePlayer[s]!))) !== s) {
        // Stale entry: the state's key was re-registered by a better path.
        // (Recompute cheaply: keys are only re-registered with a lower g.)
        continue;
      }
      const boxes = stateBoxes[s]!;
      if (this.solved(boxes)) {
        return this.renderPath(stateBoxes, statePlayer, parent, pushFrom, pushDir, s);
      }
      let stop = false;
      this.forEachPush(boxes, statePlayer[s]!, (next, from, dir) => {
        const low = this.regionLow(from);
        const key = this.keyOf(next, low);
        const prev = seenKeys.get(key);
        if (prev !== undefined && stateG[prev]! <= g + 1) return true;
        if (stateBoxes.length >= cap) {
          stop = true;
          return false;
        }
        const idx = stateBoxes.length;
        seenKeys.set(key, idx);
        stateBoxes.push(next);
        statePlayer.push(from);
        stateG.push(g + 1);
        parent.push(s);
        pushFrom.push(from);
        pushDir.push(dir);
        heapPush([g + 1 + assignmentBound(next, this.dist, b.goals), g + 1, idx]);
        return true;
      });
      if (stop) return null;
      for (const x of boxes) this.occupied[x] = 0;
    }
    return null;
  }

  // Lowest reachable cell for a state, without disturbing the search's
  // stamp bookkeeping used by forEachPush (recomputes from the player cell).
  regionLowForKey(boxes: number[], player: number): number {
    for (const x of boxes) this.occupied[x] = 1;
    const low = this.regionLow(player);
    for (const x of boxes) this.occupied[x] = 0;
    return low;
  }

  // Greedy best-first search (heuristic only): not optimal, but finds a
  // solving answer when the optimal searches run out of budget.
  greedy(cap: number): string | null {
    const { b } = this;
    for (const x of b.boxes) if (!this.live[x]) return null;
    if (this.solved(b.boxes)) return '';
    const stateBoxes: number[][] = [b.boxes.slice()];
    const statePlayer: number[] = [b.player];
    const parent: number[] = [-1];
    const pushFrom: number[] = [-1];
    const pushDir: number[] = [-1];
    const seenKeys = new Map<string, number>();
    for (const x of b.boxes) this.occupied[x] = 1;
    const low0 = this.regionLow(b.player);
    for (const x of b.boxes) this.occupied[x] = 0;
    seenKeys.set(this.keyOf(b.boxes, low0), 0);
    const heap: [number, number][] = []; // [h, state]
    const heapPush = (e: [number, number]): void => {
      let i = heap.length;
      heap.push(e);
      while (i > 0) {
        const p = (i - 1) >> 1;
        if (heap[i]![0] > heap[p]![0] || (heap[i]![0] === heap[p]![0] && heap[i]![1] >= heap[p]![1])) break;
        const t = heap[i]!;
        heap[i] = heap[p]!;
        heap[p] = t;
        i = p;
      }
    };
    const heapPop = (): [number, number] => {
      const top = heap[0]!;
      const last = heap.pop()!;
      if (heap.length > 0) {
        heap[0] = last;
        for (let i = 0; ; ) {
          let c = 2 * i + 1;
          if (c >= heap.length) break;
          if (c + 1 < heap.length && (heap[c + 1]![0] < heap[c]![0] || (heap[c + 1]![0] === heap[c]![0] && heap[c + 1]![1] < heap[c]![1]))) c++;
          if (heap[c]![0] > heap[i]![0] || (heap[c]![0] === heap[i]![0] && heap[c]![1] >= heap[i]![1])) break;
          const t = heap[i]!;
          heap[i] = heap[c]!;
          heap[c] = t;
          i = c;
        }
      }
      return top;
    };
    heapPush([assignmentBound(b.boxes, this.dist, b.goals), 0]);
    const expanded = new Set<number>();
    while (heap.length > 0) {
      const [, s] = heapPop();
      if (expanded.has(s)) continue;
      expanded.add(s);
      const boxes = stateBoxes[s]!;
      if (this.solved(boxes)) {
        return this.renderPath(stateBoxes, statePlayer, parent, pushFrom, pushDir, s);
      }
      let stop = false;
      this.forEachPush(boxes, statePlayer[s]!, (next, from, dir) => {
        const low = this.regionLow(from);
        const key = this.keyOf(next, low);
        if (seenKeys.has(key)) return true;
        if (stateBoxes.length >= cap) {
          stop = true;
          return false;
        }
        const idx = stateBoxes.length;
        seenKeys.set(key, idx);
        stateBoxes.push(next);
        statePlayer.push(from);
        parent.push(s);
        pushFrom.push(from);
        pushDir.push(dir);
        heapPush([assignmentBound(next, this.dist, b.goals), idx]);
        return true;
      });
      if (stop) return null;
      for (const x of boxes) this.occupied[x] = 0;
    }
    return null;
  }

  solve(): string | null {
    return this.bfs(BFS_STATE_CAP) ?? this.astar(ASTAR_STATE_CAP) ?? this.greedy(GREEDY_STATE_CAP);
  }
}

// Split a level into rooms: cut the hallway row's dead squares (a box there
// reaches no goal) out of the floor; each remaining part holding boxes is a
// room. Null when the level is no set of rooms.
function roomsOf(level: Board): { squares: number[]; boxes: number[]; goals: number[] }[] | null {
  const live = liveSquares(level);
  const row = Math.floor(level.player / level.w);
  const cut = new Uint8Array(level.w * level.h);
  for (let i = row * level.w; i < (row + 1) * level.w; i++) {
    if (!level.wall[i] && !live[i]) cut[i] = 1;
  }
  if (level.boxes.some((x) => cut[x])) return null;
  const part = new Int32Array(level.w * level.h).fill(-1);
  const rooms: { squares: number[]; boxes: number[]; goals: number[] }[] = [];
  for (let s = 0; s < level.w * level.h; s++) {
    if (level.wall[s] || cut[s] || part[s] >= 0) continue;
    const squares = [s];
    part[s] = rooms.length;
    for (let i = 0; i < squares.length; i++) {
      for (let k = 0; k < 4; k++) {
        const q = squares[i]! + DY[k]! * level.w + DX[k]!;
        if (q < 0 || q >= part.length) continue;
        if (!level.wall[q] && !cut[q] && part[q] < 0) {
          part[q] = rooms.length;
          squares.push(q);
        }
      }
    }
    rooms.push({ squares, boxes: [], goals: [] });
  }
  for (const x of level.boxes) rooms[part[x]!]!.boxes.push(x);
  for (const g of level.goals) {
    if (part[g]! < 0) return null;
    rooms[part[g]!]!.goals.push(g);
  }
  if (rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

// Simulate moves on a board; returns the player's final cell.
function endCell(b: Board, from: number, moves: string): number {
  let p = from;
  for (const ch of moves) {
    const c = ch.toLowerCase();
    const k = STEP_CH.indexOf(c);
    p += DY[k]! * b.w + DX[k]!;
  }
  return p;
}

function solveLevel(text: string): string {
  const level = parseLevel(text);
  const rooms = roomsOf(level);
  if (rooms === null) {
    return new PushSearch(level).solve() ?? '';
  }
  let player = level.player;
  let moves = '';
  const done: boolean[] = rooms.map(() => false);
  for (let r = 0; r < rooms.length; r++) {
    const room = rooms[r]!;
    // Other rooms' boxes stand as walls (on their goals once done).
    const wall = level.wall.slice();
    for (let o = 0; o < rooms.length; o++) {
      if (o === r) continue;
      const other = rooms[o]!;
      for (const x of done[o] ? other.goals : other.boxes) wall[x] = 1;
    }
    const sub: Board = {
      w: level.w,
      h: level.h,
      wall,
      goals: room.goals.slice().sort((a, b) => a - b),
      boxes: room.boxes.slice().sort((a, b) => a - b),
      player,
    };
    const solved = new PushSearch(sub).solve();
    if (solved === null) return moves; // unsolved room: keep what we have
    moves += solved;
    player = endCell(level, player, solved);
    done[r] = true;
  }
  return moves;
}

for (const id of readdirSync(IN_DIR).sort()) {
  const moves = solveLevel(readFileSync(join(IN_DIR, id, 'level.txt'), 'utf8'));
  mkdirSync(join(OUT_DIR, id), { recursive: true });
  writeFileSync(join(OUT_DIR, id, 'solution.txt'), `${moves}\n`);
}
