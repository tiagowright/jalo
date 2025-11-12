from hardware import KeyboardHardware
from layout import KeyboardLayout, LayoutKey
from typing import List
from numba import jit
import numba.typed
from freqdist import FreqDist, NgramType
from metrics import Metric, ObjectiveFunction
import numpy as np

class KeyboardModel:
    """
    Keyboard layout scoring model with pre-aggregated metrics and fast swap deltas.
    """

    def __init__(
        self,
        hardware: KeyboardHardware,
        metrics: List[Metric],
        objective: ObjectiveFunction,
        freqdist: FreqDist
    ):
        self.hardware = hardware
        self.freqdist = freqdist
        self.metrics = metrics
        self.objective = objective

        for metric in self.objective.metrics:
            if metric not in self.metrics:
                self.metrics.append(metric)
        
        self._metrics_for_all_position_ngrams()
        self._preaggregate_objective()

    def _metrics_for_all_position_ngrams(self):
        '''
        Pre-aggregate all metrics into position n-grams.
        M[metric] = np_array(shape=(N,)) for order 1 metrics
          M[metric][i] = metric.function(self.hardware.positions[i])
        M[metric] = np_array(shape=(N,N)) for order 2 metrics
          M[metric][i,j] = metric.function(self.hardware.positions[i], self.hardware.positions[j])
        M[metric] = np_array(shape=(N,N,N)) for order 3 metrics
          M[metric][i,j,k] = metric.function(self.hardware.positions[i], self.hardware.positions[j], self.hardware.positions[k])
        '''
        self.M = {}

        for metric in self.metrics:
            if metric.order == 1:
                self.M[metric] = np.array(
                    [
                        metric.function(self.hardware.positions[i]) for i in range(len(self.hardware.positions))  # pyright: ignore[reportCallIssue]
                    ], dtype=np.float64
                )
            elif metric.order == 2:
                self.M[metric] = np.array(
                    [
                        [
                            metric.function(self.hardware.positions[i], self.hardware.positions[j])  # pyright: ignore[reportCallIssue]
                            for j in range(len(self.hardware.positions))
                        ] for i in range(len(self.hardware.positions))
                    ], dtype=np.float64
                )
            elif metric.order == 3:
                self.M[metric] = np.array(
                    [
                        [
                            [
                                metric.function(self.hardware.positions[i], self.hardware.positions[j], self.hardware.positions[k])  # pyright: ignore[reportCallIssue]
                                for k in range(len(self.hardware.positions))
                            ] for j in range(len(self.hardware.positions))
                        ] for i in range(len(self.hardware.positions))
                    ], dtype=np.float64
                )

    def _preaggregate_objective(self):
        '''
        Pre-aggregate all metrics into a single tensor for the objective function.

        requires self.M to be computed first. Then we compute the linear combination
        of each self.M[metric] for each ngramtype, into tensors of the shape that corresponds
        to the ngramtype (order 1 will be (N,), order 2 will be (N,N), order 3 will be (N,N,N)).
        
        To compute the final score, when the char positions are known, we simply aggregate the
        position frequency tensors * the pre-aggregated objective tensors, with appropriate weights.
        '''
        N = len(self.hardware.positions)

        # shape is (N,N,N) for ngramtype.order == 3, (N,N) for ngramtype.order == 2, (N,) for ngramtype.order == 1
        self.V = {
            ngramType : np.zeros(
                (N,N,N) if ngramType.order == 3 else 
                (N,N) if ngramType.order == 2 else 
                (N,), 
                dtype=np.float64
            ) for ngramType in NgramType if ngramType.order <= 3
        }

        for metric, weight in self.objective.metrics.items():
            self.V[metric.ngramType] += weight * self.M[metric]
        

    def position_freqdist(self, char_at_pos: np.ndarray) -> dict[NgramType, np.ndarray]:
        '''
        Convert character ngram frequencies into position ngram frequencies for the layout specified by char_at_pos.
        '''
        F = self.freqdist.to_numpy()
        C = np.asarray(char_at_pos, dtype=int)
        P = {}
        for ngramtype in F:
            if ngramtype.order == 1:
                P[ngramtype] = F[ngramtype][C]
            elif ngramtype.order == 2:
                P[ngramtype] = F[ngramtype][np.ix_(C, C)]
            elif ngramtype.order == 3:
                P[ngramtype] = F[ngramtype][np.ix_(C, C, C)]
        return P

    def analyze_chars_at_positions(self, char_at_pos: np.ndarray) -> dict[Metric, float]:
        '''
        Analyze the characters at positions specified by char_at_pos.
        '''
        P = self.position_freqdist(char_at_pos)
        return {metric: float(np.sum(P[metric.ngramType] * self.M[metric])) for metric in self.metrics}

    def analyze_layout(self, layout: KeyboardLayout) -> dict[Metric, float]:
        '''
        Analyze the layout specified by the KeyboardLayout.
        '''
        char_at_pos = self.char_at_positions_from_layout(layout)
        return self.analyze_chars_at_positions(char_at_pos)
    
    def score_chars_at_positions(self, char_at_pos: np.ndarray | tuple[int, ...]) -> float:
        '''
        Score the characters at positions specified by char_at_pos.
        '''
        P = self.position_freqdist(char_at_pos)
        return sum(
            float(np.sum(P[ngramtype] * self.V[ngramtype]))
            for ngramtype in P 
            if ngramtype in self.V
        )

    def score_layout(self, layout: KeyboardLayout) -> float:
        '''
        Score the layout specified by the KeyboardLayout.
        '''
        char_at_pos = self.char_at_positions_from_layout(layout)
        return self.score_chars_at_positions(char_at_pos)

    def score_contributions(self, layout: KeyboardLayout) -> dict[Metric, float]:
        '''
        Score the contributions of each metric to the total score.
        '''
        char_at_pos = self.char_at_positions_from_layout(layout)
        P = self.position_freqdist(char_at_pos)
        
        return {
            metric: float(np.sum(P[metric.ngramType] * self.objective.metrics[metric] * self.M[metric]))
            if metric in self.objective.metrics and metric in self.M and metric.ngramType in P else 0.0
            for metric in self.metrics
        }

    def char_at_positions_from_layout(self, layout: KeyboardLayout) -> np.ndarray:
        # validate that layout uses the same hardware
        if layout.hardware != self.hardware:
            raise ValueError(f"Layout hardware {layout.hardware.name} does not match model hardware {self.hardware.name}")
        
        char_at_pos = np.zeros(len(self.hardware.positions), dtype=int)
        for pi, position in enumerate(self.hardware.positions):
            key = layout.key_at_position[position]
            try:
                char_at_pos[pi] = self.freqdist.char_seq.index(key.char)
            except ValueError:
                char_at_pos[pi] = self.freqdist.char_seq.index(FreqDist.out_of_distribution)
        return char_at_pos

    def layout_from_char_at_positions(self, char_at_pos: np.ndarray | tuple[int, ...], original_layout: KeyboardLayout | None = None) -> KeyboardLayout:
        '''
        Convert the character at positions specified by char_at_pos into a KeyboardLayout.
        '''
        assert len(char_at_pos) == len(self.hardware.positions)
        assert original_layout is None or original_layout.hardware == self.hardware

        name = ''
        keys = []
        for pi, ci in enumerate(char_at_pos):
            char = self.freqdist.char_seq[ci]
            if char == FreqDist.out_of_distribution:
                if original_layout:
                    char = original_layout.char_at_position[self.hardware.positions[pi]]
                else:
                    char = ''
            if self.hardware.positions[pi].is_home:
                name += char
            keys.append(LayoutKey.from_position(self.hardware.positions[pi], char))
        return KeyboardLayout(keys, self.hardware, name if not original_layout else original_layout.name)

    def calculate_swap_delta(self, char_at_pos: np.ndarray, i: int, j: int) -> float:
        F = self.freqdist.to_numpy()

        order_1 = tuple([
            (F[ngramtype], self.V[ngramtype])
            for ngramtype in NgramType 
            if ngramtype.order == 1 and ngramtype in F and ngramtype in self.V
        ])
        order_2 = tuple([
            (F[ngramtype], self.V[ngramtype]) 
            for ngramtype in NgramType 
            if ngramtype.order == 2 and ngramtype in F and ngramtype in self.V
        ])
        order_3 = tuple([
            (F[ngramtype], self.V[ngramtype]) 
            for ngramtype in NgramType 
            if ngramtype.order == 3 and ngramtype in F and ngramtype in self.V
        ])

        new_char_at_pos = tuple(
            char_at_pos[j] if k == i else
            char_at_pos[i] if k == j else
            char_at_pos[k]
            for k in range(len(char_at_pos))
        )

        return _calculate_swap_delta(order_1, order_2, order_3, tuple(char_at_pos), i, j, new_char_at_pos)  # pyright: ignore[reportArgumentType]

@jit
def _calculate_swap_delta(
    order_1: tuple[tuple[np.ndarray, np.ndarray]],  
    order_2: tuple[tuple[np.ndarray, np.ndarray]],  
    order_3: tuple[tuple[np.ndarray, np.ndarray]],  
    char_at_pos: tuple[int],  
    i: int,
    j: int,
    new_char_at_pos: tuple[int]
) -> float:
    """
    Calculates the change in score from swapping char_at_pos[i] and char_at_pos[j].
    A negative delta means the new layout is better (lower score).
    
    This is an O(n^2) operation, much faster than a full O(n^3) rescore.

    This version of the function is optimized for JIT compilation with Numba, to accelerate the calculation of the swap delta.
    """
    if i == j:
        return 0.0

    C = char_at_pos
    N = len(C)
    
    # Character indices at positions i and j
    c_i = C[i]
    c_j = C[j]

    # The 'new' layout, C_prime, is implied:
    # C_prime[i] = c_j
    # C_prime[j] = c_i
    C_prime = new_char_at_pos

    # We will calculate delta_E = new_score - old_score
    # For all terms that are affected by the swap.
    delta_E = 0.0

    # ---
    # 1. Unigram (Order 1) Delta - O(1)
    # ---
    for F1, V1 in order_1:
        old_score_1 = (F1[c_i] * V1[i]) + (F1[c_j] * V1[j])
        new_score_1 = (F1[c_j] * V1[i]) + (F1[c_i] * V1[j])
        delta_E += new_score_1 - old_score_1

    # ---
    # 2. Bigram and Skipgram (Order 2) Delta - O(n)
    # ---
    for F2, V2 in order_2:
        if F2 is None or V2 is None:
            continue

        old_score_2 = 0.0
        new_score_2 = 0.0

        for k in range(N):
            c_k = C[k]

            if k == i:
                c_prime_k = c_j
            elif k == j:
                c_prime_k = c_i
            else:
                c_prime_k = c_k

            old_score_2 += (F2[c_i, c_k] * V2[i, k]) + (F2[c_k, c_i] * V2[k, i])
            new_score_2 += (F2[c_j, c_prime_k] * V2[i, k]) + (F2[c_prime_k, c_j] * V2[k, i])

            old_score_2 += (F2[c_j, c_k] * V2[j, k]) + (F2[c_k, c_j] * V2[k, j])
            new_score_2 += (F2[c_i, c_prime_k] * V2[j, k]) + (F2[c_prime_k, c_i] * V2[k, j])

        old_score_2 -= (F2[c_i, c_i] * V2[i, i]) + (F2[c_i, c_j] * V2[i, j]) + (F2[c_j, c_i] * V2[j, i]) + (F2[c_j, c_j] * V2[j, j])
        new_score_2 -= (F2[c_j, c_j] * V2[i, i]) + (F2[c_j, c_i] * V2[i, j]) + (F2[c_i, c_j] * V2[j, i]) + (F2[c_i, c_i] * V2[j, j])

        delta_E += new_score_2 - old_score_2

    # ---
    # 3. Trigram (Order 3) Delta - O(n^2)
    # ---
    for F3, V3 in order_3:

        for k in range(N):
            for l in range(N):
                c_k, c_l = C[k], C[l]
                c_prime_k, c_prime_l = C_prime[k], C_prime[l]

                old_term = F3[c_i, c_k, c_l] * V3[i, k, l]
                new_term = F3[c_j, c_prime_k, c_prime_l] * V3[i, k, l]
                delta_E += new_term - old_term

                old_term = F3[c_k, c_i, c_l] * V3[k, i, l]
                new_term = F3[c_prime_k, c_j, c_prime_l] * V3[k, i, l]
                delta_E += new_term - old_term

                old_term = F3[c_k, c_l, c_i] * V3[k, l, i]
                new_term = F3[c_prime_k, c_prime_l, c_j] * V3[k, l, i]
                delta_E += new_term - old_term

                old_term = F3[c_j, c_k, c_l] * V3[j, k, l]
                new_term = F3[c_i, c_prime_k, c_prime_l] * V3[j, k, l]
                delta_E += new_term - old_term

                old_term = F3[c_k, c_j, c_l] * V3[k, j, l]
                new_term = F3[c_prime_k, c_i, c_prime_l] * V3[k, j, l]
                delta_E += new_term - old_term

                old_term = F3[c_k, c_l, c_j] * V3[k, l, j]
                new_term = F3[c_prime_k, c_prime_l, c_i] * V3[k, l, j]
                delta_E += new_term - old_term

        for k in range(N):
            c_k = C[k]
            c_prime_k = C_prime[k]

            old_term = F3[c_i, c_i, c_k] * V3[i, i, k]
            new_term = F3[c_j, c_j, c_prime_k] * V3[i, i, k]
            delta_E -= new_term - old_term

            old_term = F3[c_i, c_k, c_i] * V3[i, k, i]
            new_term = F3[c_j, c_prime_k, c_j] * V3[i, k, i]
            delta_E -= new_term - old_term

            old_term = F3[c_k, c_i, c_i] * V3[k, i, i]
            new_term = F3[c_prime_k, c_j, c_j] * V3[k, i, i]
            delta_E -= new_term - old_term

            old_term = F3[c_j, c_j, c_k] * V3[j, j, k]
            new_term = F3[c_i, c_i, c_prime_k] * V3[j, j, k]
            delta_E -= new_term - old_term

            old_term = F3[c_j, c_k, c_j] * V3[j, k, j]
            new_term = F3[c_i, c_prime_k, c_i] * V3[j, k, j]
            delta_E -= new_term - old_term

            old_term = F3[c_k, c_j, c_j] * V3[k, j, j]
            new_term = F3[c_prime_k, c_i, c_i] * V3[k, j, j]
            delta_E -= new_term - old_term

            old_term = F3[c_i, c_j, c_k] * V3[i, j, k]
            new_term = F3[c_j, c_i, c_prime_k] * V3[i, j, k]
            delta_E -= new_term - old_term

            old_term = F3[c_i, c_k, c_j] * V3[i, k, j]
            new_term = F3[c_j, c_prime_k, c_i] * V3[i, k, j]
            delta_E -= new_term - old_term

            old_term = F3[c_j, c_i, c_k] * V3[j, i, k]
            new_term = F3[c_i, c_j, c_prime_k] * V3[j, i, k]
            delta_E -= new_term - old_term

            old_term = F3[c_j, c_k, c_i] * V3[j, k, i]
            new_term = F3[c_i, c_prime_k, c_j] * V3[j, k, i]
            delta_E -= new_term - old_term

            old_term = F3[c_k, c_i, c_j] * V3[k, i, j]
            new_term = F3[c_prime_k, c_j, c_i] * V3[k, i, j]
            delta_E -= new_term - old_term

            old_term = F3[c_k, c_j, c_i] * V3[k, j, i]
            new_term = F3[c_prime_k, c_i, c_j] * V3[k, j, i]
            delta_E -= new_term - old_term

        old_term = F3[c_i, c_i, c_i] * V3[i, i, i]
        new_term = F3[c_j, c_j, c_j] * V3[i, i, i]
        delta_E += new_term - old_term

        old_term = F3[c_j, c_j, c_j] * V3[j, j, j]
        new_term = F3[c_i, c_i, c_i] * V3[j, j, j]
        delta_E += new_term - old_term

        old_term = F3[c_i, c_i, c_j] * V3[i, i, j]
        new_term = F3[c_j, c_j, c_i] * V3[i, i, j]
        delta_E += new_term - old_term

        old_term = F3[c_i, c_j, c_i] * V3[i, j, i]
        new_term = F3[c_j, c_i, c_j] * V3[i, j, i]
        delta_E += new_term - old_term

        old_term = F3[c_j, c_i, c_i] * V3[j, i, i]
        new_term = F3[c_i, c_j, c_j] * V3[j, i, i]
        delta_E += new_term - old_term

        old_term = F3[c_j, c_j, c_i] * V3[j, j, i]
        new_term = F3[c_i, c_i, c_j] * V3[j, j, i]
        delta_E += new_term - old_term

        old_term = F3[c_j, c_i, c_j] * V3[j, i, j]
        new_term = F3[c_i, c_j, c_i] * V3[j, i, j]
        delta_E += new_term - old_term

        old_term = F3[c_i, c_j, c_j] * V3[i, j, j]
        new_term = F3[c_j, c_i, c_i] * V3[i, j, j]
        delta_E += new_term - old_term

    return delta_E


if __name__ == "__main__":
    import argparse
    from freqdist import FreqDist
    from metrics import METRICS
    from hardware import KeyboardHardware
    from layout import KeyboardLayout

    parser = argparse.ArgumentParser(description='Keyboard model metrics and position frequency distribution')
    parser.add_argument('-m', '--metrics', action='store_true', help='Output metric data')
    parser.add_argument('-p', '--position-freqdist', action='store_true', help='Output position frequency distribution (P)')
    parser.add_argument('-a', '--analyze-layout', action='store_true', help='Analyze the layout')
    parser.add_argument('-l', '--layout', default='qwerty', help='Layout file to use for computing P (default: qwerty.kb)')
    parser.add_argument('-o', '--objective', action='store_true', help='output the pre-aggregated objective function V')
    parser.add_argument('-hw', '--hardware', default='ortho', help='Hardware to use for computing P (default: ortho)')
    
    args = parser.parse_args()
    
    # If no flags provided, default to -m for backward compatibility
    if not args.metrics and not args.position_freqdist and not args.analyze_layout and not args.objective:
        args.metrics = True

    freqdist = FreqDist.from_name("en")
    metrics = METRICS
    hardware = KeyboardHardware.from_name(args.hardware)
    objective = ObjectiveFunction({metrics[0]: 2.0, metrics[2]: 1.5, metrics[3]: 3.0, metrics[18]: 1.1})
    model = KeyboardModel(hardware, metrics, objective, freqdist)
    layout = KeyboardLayout.from_name(args.layout, hardware)

    if args.metrics:
        for metric in model.M:
            print(metric.name)
            if metric.order == 1:
                print(model.M[metric])
            elif metric.order == 2:
                # print each row of M[metric] in a separate line, print ints separated by 1 space
                # print the row index on the left of each taking 3 characters wide row then a ':'
                for i, row in enumerate(model.M[metric]):
                    print(f'{i:3d}: ', end='')
                    print(' '.join(str(int(x)) for x in row))

            elif metric.order == 3:
                for j, plane in enumerate(model.M[metric]):
                    for i, row in enumerate(plane):
                        print(f'{j:3d},{i:3d}: ', end='')
                        print(' '.join(str(int(x)) for x in row))

            print()

    if args.position_freqdist:
        # Compute P
        P = model.position_freqdist(model.char_at_positions_from_layout(layout))
        
        # Output P
        for ngramtype in P:
            print(f"{ngramtype.name}:")
            if ngramtype.order == 1:
                print(P[ngramtype])
            elif ngramtype.order == 2:
                for i, row in enumerate(P[ngramtype]):
                    print(f'{i:3d}: ', end='')
                    print(' '.join(f'{x:.6f}' for x in row))
            elif ngramtype.order == 3:
                for j, plane in enumerate(P[ngramtype]):
                    for i, row in enumerate(plane):
                        print(f'{j:3d},{i:3d}: ', end='')
                        print(' '.join(f'{x:.6f}' for x in row))
            print()

    if args.analyze_layout:
        # print(model.analyze_layout(layout))
        analysis = model.analyze_layout(layout)
        for metric in analysis:
            print(f"{metric.name:13s}: {analysis[metric]*100:>8.4f}%")
        print()


    if args.objective:
        print(f"Objective function metrics: {model.objective.metrics}")
        print("Objective function V:")
        for ngramtype in model.V:
            print(f"{ngramtype.name}:")
            if ngramtype.order == 1:
                print(model.V[ngramtype])
            elif ngramtype.order == 2:
                for i, row in enumerate(model.V[ngramtype]):
                    print(f'{i:3d}: ', end='')
                    print(' '.join(f'{x:.6f}' for x in row))
            elif ngramtype.order == 3:
                for j, plane in enumerate(model.V[ngramtype]):
                    for i, row in enumerate(plane):
                        print(f'{j:3d},{i:3d}: ', end='')
                        print(' '.join(f'{x:.6f}' for x in row))
                    print()
            print()
