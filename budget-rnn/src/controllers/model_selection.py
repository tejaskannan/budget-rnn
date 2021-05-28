import numpy as np
from typing import List, Set

from .model_controllers import AdaptiveController
from .controller_utils import ModelResults


class AdaptiveSelector:

    def __init__(self, controllers: List[AdaptiveController], model_valid_results: List[ModelResult]):
        self._controllers = controllers
        self._valid_results = valid_results

        # Get the budgets for all controllers
        budget_set: Set[float] = set()
        for controller in controllers:
            budget_set.extend(controller.budgets)

        budgets = np.array(list(sorted(budget_set)))
        self._budgets = budgets
        
        # Create the policy
        self._budget_dict: Dict[float, AdaptiveController] = dict()  # Holds budget to model index map
        for budget in budgets:

            max_accuracy = 0.0
            best_controller = None

            for controller, valid_results in zip(controllers, model_valid_results):
                if budget in controller.budgets:
                    accuracy = controller.get_accuracy(budget=budget, model_results=valid_results)
        
                    if accuracy > max_accuracy:
                        max_accuracy = accuracy
                        best_controller = controller

            assert best_controller is not None, 'Could not find controller for budget: {0}'.format(budget)
            self._budget_dict[budget] = best_controller

    def get_controller(self, budget: float) -> AdaptiveController:
        budget_diff = np.abs(self._budgets - budget)
        model_idx = np.argmin(budget_diff)
        nearest_budget = self._budgets[budget]
        return self._budget_dict[nearest_budget]
