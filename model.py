from hardware import KeyboardHardware
from layout import KeyboardLayout
from typing import List, Tuple, Optional
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
    
    def score_chars_at_positions(self, char_at_pos: np.ndarray) -> float:
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
