"""
Dynamic layer ablation hooks.

Register on all layers. Per forward pass, one layer can be:
- ablated (output = input, skipping the layer's computation)
- doubled (layer runs twice)
- scaled (output contribution multiplied by a factor)
- normal (no modification)

Thread-safe for single-GPU training. Change target between forward passes.
"""

import torch
import torch.nn as nn


class AblationManager:
    """
    Manages per-layer hooks for dynamic ablation during training.

    Usage:
        mgr = AblationManager(model)
        mgr.register()

        # For each training step:
        mgr.set(layer=17, mode="ablate")
        loss = model(input_ids, labels=labels).loss
        loss.backward()

        mgr.set(layer=None)  # reset to normal
        mgr.remove()  # cleanup
    """

    MODES = ("ablate", "double", "scale", None)

    def __init__(self, model):
        self.model = model
        self.n_layers = len(model.model.layers)
        self.target_layer = None
        self.mode = None  # "ablate", "double", "scale"
        self.scale_factor = 1.0  # for mode="scale"
        self._hooks = []

    def register(self):
        for idx in range(self.n_layers):
            layer = self.model.model.layers[idx]
            h = layer.register_forward_hook(self._make_hook(idx))
            self._hooks.append(h)

    def _make_hook(self, layer_idx):
        def hook_fn(module, input, output):
            if layer_idx != self.target_layer or self.mode is None:
                return output

            inp = input[0] if isinstance(input, tuple) else input

            if self.mode == "ablate":
                # Skip layer: return input unchanged
                if isinstance(output, tuple):
                    return (inp,) + output[1:]
                return inp

            elif self.mode == "scale":
                # Scale the layer's contribution
                # output = input + layer_contribution
                # scaled = input + scale * layer_contribution
                contribution = (output[0] if isinstance(output, tuple) else output) - inp
                scaled = inp + self.scale_factor * contribution
                if isinstance(output, tuple):
                    return (scaled,) + output[1:]
                return scaled

            elif self.mode == "double":
                # Run layer again on its output (approximate doubling)
                # We can't easily re-run the module, so we double the contribution
                contribution = (output[0] if isinstance(output, tuple) else output) - inp
                doubled = (output[0] if isinstance(output, tuple) else output) + contribution
                if isinstance(output, tuple):
                    return (doubled,) + output[1:]
                return doubled

            return output
        return hook_fn

    def set(self, layer=None, mode="ablate", scale_factor=0.0):
        """Set which layer to modify and how."""
        self.target_layer = layer
        self.mode = mode if layer is not None else None
        self.scale_factor = scale_factor

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
