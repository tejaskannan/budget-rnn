# Controllers
This module contains the runtime controllers necessary to simulate budgeted inference. This document walks through the key components of this module.

## Model Controllers
The ``model_controllers.py` file implements two features. The first is the algorithms for controlling the number of samples consumed by an RNN during inference. The code includes the four selection algorithms below.
1. `AdaptiveController`: This class performs subsequence level selection for Budget RNNs. The selection algorithm uses tuned thresholds on the halting signals. For unknown budgets, the algorithm will create thresholds using a linear interpolation process. The function `get_budget_interpolation_values` performs the linear interpolation. 
2. `FixedController`: This class implements level selection with a fixed index. That is, during inference, this selector will always choose the same subsequence granularity. Standard RNNs use this selector.
3. `RandomController`: This class selects subsequence levels randomly. The algorithm creates subsequence selection probabilities by matching the estimated energy with that of the budget. The object fits these probabilities using SLSQP (see `power_distribution.py`). The Budget RNN randomized variant uses this selector.
4. `MultiModelController`: This class selects a single model from a collection of RNNs. Each RNN in this collection operates at a different subsequence granularity. The selection algorithm is the same as that of the fixed controller. The difference is that selection involves picking a distinct model instead of an early-exit point. Both Skip RNNs and Phased RNNs use this controller.

The `BudgetWrapper` class applies these controllers to monitor the remaining energy budget. The wrapper protects against budget violations. Further, this class will compute the sytem's prediction for a given time step. As a note, the predictions are pre-computed in a batched manner for efficiency.

The second portion of the `model_controllers.py` file involves optimizing the halting thresholds. The `BudgetOptimizer` class implements the coordinate descent algorithm used to fit halting thresholds. The key methods involved in optimization are `fit()`, `fit_single()`, and `loss_function()`. Furthermore, the optimizer uses the helper function `levels_to_execute` to find the subsequence level for a given set of thresholds and stop probabilities. The function operates on batches for efficiency.

## Runtime System
The `runtime_system.py` file implements a class that manages RNN execution during budgeted inference. For a given RNN model type, the object selects the appropriate controller. For each sequence, the runtime system will use the model controller to choose the subsequence level; this level governs the current prediction. As a note, the class precomputes predictions for efficiency.

## Runtime Controllers
The file `runtime_controllers.py` holds the Budget RNN runtime control systems. The `PIDController` and `BudgetController` classes implement the controller that sets the estimated budget. The resulting value serves as the target when interpolating halting thresholds. The `BudgetDistribution` class creates the setpoint for the PID controller. The class implements this behavior by estimating the remaining energy consumption using the observed energy thus far. The setpoint is a confidence bound, and the PID controller considers any value within this range to have zero error.

## Power System
The file `power_utils.py` implements the simulated sensors. This code measures the energy consumption of each sensor using profiled values from a TI MSP430 FR5994. Each simulated sensor computes its energy consumption as the sum of sensing and computation energy. The computation energy varies based on the RNN type. For example, Budget RNNs consume more computation energy than standard RNNs.

## Noise Generation
The `noise_generators.py` file implements classes to generate noise in the energy consumption. For efficiency reasons, the code pre-generates the noise terms for all sequences. The paper only evaluates cases with Gaussian noise. 
