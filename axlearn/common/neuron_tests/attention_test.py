import contextlib

# pylint: disable=too-many-lines,duplicate-code,no-self-use

import jax
import pytest
import numpy as np
from jax import numpy as jnp
from jax.sharding import PartitionSpec
from jax.sharding import NamedSharding
from axlearn.common import attention, test_utils, utils
from axlearn.common.attention import (
    ParallelTransformerLayer,
    TransformerLayer, scaled_hidden_dim, TransformerFeedForwardLayer, MultiheadAttention, FusedQKVLinear,
)
from axlearn.common.base_layer import ParameterSpec
from axlearn.common.layers import RMSNorm, set_bias_recursively
from axlearn.common.module import functional as F
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

from axlearn.common.utils import Tensor, VDict
from axlearn.common.test_utils import NeuronTestCase, assert_allclose, dummy_segments_positions


def backup(self):
    self.assertEqual(
        {
            "feed_forward": {
                "dropout1": {},
                "dropout2": {},
                "linear1": {"weight": (16, 64)},
                "linear2": {"weight": (64, 16)},
                "stochastic_depth": {},
            },
            "norm": {"scale": (16,)},
            "self_attention": {
                "dropout": {},
                "i_proj": {
                    "k_proj": {"weight": (16, 4, 4)},
                    "q_proj": {"weight": (16, 4, 4)},
                    "v_proj": {"weight": (16, 4, 4)},
                },
                "o_proj": {"weight": (16, 4, 4)},
                "scale_key": {},
                "scale_query": {},
            },
        },
        #utils.shapes(layer_params),
        utils.shapes(''),
    )
class TransformerTest(NeuronTestCase):
    """Tests TransformerLayer."""

    #@pytest.mark.skip
    def test_forward(self):
        """A test of TransformerLayer forward."""
        mesh = jax.sharding.Mesh(np.array(jax.devices()).reshape(4, 8)[:, None, None, None, :],
                                 axis_names=("data", "seq", "expert", "fsdp", "model"),)
        with mesh:
            model_dim = 16 
            num_heads = 8 
            cfg = TransformerLayer.default_config().set(name="test", input_dim=model_dim)
            cfg.feed_forward.set(hidden_dim=scaled_hidden_dim(4))
            cfg.self_attention.attention.set(num_heads=num_heads)
            cfg.self_attention.attention.input_linear = FusedQKVLinear.default_config()
            cfg.self_attention.norm = RMSNorm.default_config()
            cfg.feed_forward.norm = RMSNorm.default_config()
            set_bias_recursively(cfg, bias=False)
            set_double_shard_weights_config(
                cfg,
                batch_axis_names='data',
                fsdp_axis_names='fsdp',
                tp_axis_names='model',
                seq_axis_names='model',
            )
            layer: TransformerLayer = cfg.instantiate(parent=None)
            self._trainer_state_specs = collect_param_specs(layer)
            def create_named_sharding(param_spec, mesh):
                if isinstance(param_spec, ParameterSpec):
                    return NamedSharding(
                        mesh,
                        PartitionSpec(*param_spec.mesh_axes) if param_spec.mesh_axes != (None,) else PartitionSpec(None)
                    )
                return param_spec

            def custom_tree_map(func, pytree, mesh):
                if isinstance(pytree, dict) or isinstance(pytree, VDict):
                    new_dict = {}
                    for k, v in pytree.items():
                        if k == 'i_proj': # Weird case where i_proj is a Vdict not a Dict
                            new_dict[k] = VDict({sub_k: custom_tree_map(func, sub_v, mesh) for sub_k, sub_v in v.items()})
                        else:
                            new_dict[k] = custom_tree_map(func, v, mesh)
                    return type(pytree)(new_dict)
                else:
                    return func(pytree)

            self._trainer_state_partition_specs = custom_tree_map(
                lambda ps: create_named_sharding(ps, mesh),
                self._trainer_state_specs,
                mesh
            )
            def init_cpu():  # Initing on Neuron causes compiler failures.
                layer_params = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(0))
                return layer_params

            def move_to_neuron(params):
                weights = jax.device_put(params)
                return weights
            cpu_device = jax.devices('cpu')[0]
            with jax.default_device(cpu_device):
                layer_params = init_cpu()

            move_to_neuron = jax.jit(
                move_to_neuron,
                in_shardings=(self._trainer_state_partition_specs,), # tuple is necessary here
            )
            layer_params = move_to_neuron(layer_params)
            def print_dict_structure(d, indent=0):
                for key, value in d.items():
                    print(' ' * indent + f"{key}: {type(value)}")
                    if isinstance(value, dict):
                        print_dict_structure(value, indent + 4)

            #print_dict_structure(layer_params)

            #jax.debug.visualize_array_sharding(layer_params['feed_forward']['linear1']['weight'])

            batch_size, tgt_len = 4, 32
            rng = np.random.default_rng(seed=123)
            target = rng.random([batch_size, tgt_len, model_dim], dtype=np.float32)
            mask = attention.make_causal_mask(tgt_len)
            mask = jnp.tile(mask[None, None, :, :], (batch_size, num_heads, 1, 1))
            mask = jax.device_put(mask, NamedSharding(mesh, PartitionSpec('data', 'model', None, None)))
            input_tensor = jnp.asarray(target)
            input_tensor = jax.device_put(input_tensor, NamedSharding(mesh, PartitionSpec('data', 'model', None)))

            def run(mask, input_tensor, weights):
                layer_outputs, _ = F(
                    layer,
                    inputs=dict(data=input_tensor, self_attention_logit_biases=mask),
                    state=weights,
                    is_training=True,
                    prng_key=jax.random.PRNGKey(0),
                )
                return layer_outputs
            run = jax.jit(run, in_shardings=(NamedSharding(mesh, PartitionSpec('data', 'model', None, None)),
                                             NamedSharding(mesh, PartitionSpec('data', 'model', None)),
                                             self._trainer_state_partition_specs))
            layer_outputs = run(mask, input_tensor, layer_params)
            self.assertEqual(target.shape, layer_outputs.data.shape)

    @pytest.mark.skip
    def test_backward(self):
        """A test of TransformerLayer backward."""
        mesh = jax.sharding.Mesh(np.array(jax.devices()).reshape(4, 8)[:, None, None, None, :],
                                 axis_names=("data", "seq", "expert", "fsdp", "model"),)
        with mesh:
            model_dim = 4096
            num_heads = 32
            cfg = TransformerLayer.default_config().set(name="test", input_dim=model_dim)
            cfg.dtype = jnp.bfloat16
            cfg.feed_forward.set(hidden_dim=scaled_hidden_dim(4))
            cfg.self_attention.attention.set(num_heads=num_heads)
            cfg.self_attention.attention.input_linear = FusedQKVLinear.default_config()
            cfg.self_attention.norm = RMSNorm.default_config()
            cfg.feed_forward.norm = RMSNorm.default_config()
            set_bias_recursively(cfg, bias=False)
            set_double_shard_weights_config(
                cfg,
                batch_axis_names='data',
                fsdp_axis_names='fsdp',
                tp_axis_names='model',
                seq_axis_names='model',
            )
            layer: TransformerLayer = cfg.instantiate(parent=None)
            self._trainer_state_specs = collect_param_specs(layer)
            def create_named_sharding(param_spec, mesh):
                if isinstance(param_spec, ParameterSpec):
                    return NamedSharding(
                        mesh,
                        PartitionSpec(*param_spec.mesh_axes) if param_spec.mesh_axes != (None,) else PartitionSpec(None)
                    )
                return param_spec

            def custom_tree_map(func, pytree, mesh):
                if isinstance(pytree, dict) or isinstance(pytree, VDict):
                    new_dict = {}
                    for k, v in pytree.items():
                        if k == 'i_proj': # Weird case where i_proj is a Vdict not a Dict
                            new_dict[k] = VDict({sub_k: custom_tree_map(func, sub_v, mesh) for sub_k, sub_v in v.items()})
                        else:
                            new_dict[k] = custom_tree_map(func, v, mesh)
                    return type(pytree)(new_dict)
                else:
                    return func(pytree)

            self._trainer_state_partition_specs = custom_tree_map(
                lambda ps: create_named_sharding(ps, mesh),
                self._trainer_state_specs,
                mesh
            )
            def init_cpu():  # Initing on Neuron causes compiler failures.
                layer_params = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(0))
                return layer_params

            def move_to_neuron(params):
                weights = jax.device_put(params)
                return weights
            cpu_device = jax.devices('cpu')[0]
            with jax.default_device(cpu_device):
                layer_params = init_cpu()

            move_to_neuron = jax.jit(
                move_to_neuron,
                in_shardings=(self._trainer_state_partition_specs,), # tuple is necessary here
            )
            layer_params = move_to_neuron(layer_params)
            def print_dict_structure(d, indent=0):
                for key, value in d.items():
                    print(' ' * indent + f"{key}: {type(value)}")
                    if isinstance(value, dict):
                        print_dict_structure(value, indent + 4)

            #print_dict_structure(layer_params)

            jax.debug.visualize_array_sharding(layer_params['feed_forward']['linear1']['weight'])
            batch_size, tgt_len = 4, 4096
            rng = np.random.default_rng(seed=123)
            target = rng.random([batch_size, tgt_len, model_dim], dtype=np.float32).astype(jnp.bfloat16)
            target = jax.device_put(target, NamedSharding(mesh, PartitionSpec('data', 'model', None)))
            def mask_creation():
                mask = attention.make_causal_mask(tgt_len).astype(jnp.bfloat16)
                mask = jnp.tile(mask[None, None, :, :], (batch_size, num_heads, 1, 1))
                mask = jax.device_put(mask)
                return mask
            mask_creation = jax.jit(mask_creation, out_shardings=NamedSharding(mesh, PartitionSpec('data', 'model', None, None)))
            mask = mask_creation()
            input_tensor = jnp.asarray(target).astype(jnp.bfloat16)
            input_tensor = jax.device_put(input_tensor, NamedSharding(mesh, PartitionSpec('data', 'model', None)))

            print(input_tensor)
            #jax.debug.visualize_array_sharding(input_tensor)
            def run(mask, input_tensor, output_target, weights):
                layer_outputs, _ = F(
                    layer,
                    inputs=dict(data=input_tensor, self_attention_logit_biases=mask),
                    state=weights,
                    is_training=True,
                    prng_key=jax.random.PRNGKey(0),
                )

                return jnp.mean((layer_outputs.data - output_target) ** 2)

            run = jax.jit(jax.value_and_grad(run), in_shardings=(NamedSharding(mesh, PartitionSpec('data', 'model', None, None)),
                                             NamedSharding(mesh, PartitionSpec('data', 'model', None)),
                                             NamedSharding(mesh, PartitionSpec('data', 'model', None)),
                                             self._trainer_state_partition_specs))
            loss, grad = run(mask, input_tensor, target, layer_params)
            print(loss)
            print(grad)



def set_double_shard_weights_config(
        cfg: Union[TransformerLayer.Config, Sequence[TransformerLayer.Config]],
        *,
        batch_axis_names: Union[str, Sequence[str]] = ("data", "fsdp"),
        fsdp_axis_names: Union[str, Sequence[str]] = "fsdp",
        tp_axis_names: Union[str, Sequence[str]] = "model",
        seq_axis_names: Union[str, Sequence[str]] = "seq",
):
    """Sets `cfg` to shard FFN and attention weights over both fsdp and tp axes.

    Args:
        cfg: (A sequence of) Transformer layer config to apply sharding spec to.
        batch_axis_names: Axis name(s) over which we shard the batch dimension of output tensors.
        fsdp_axis_names: Axis name(s) over which we shard fully-sharded-data-parallel tensors.
        tp_axis_names: Axis name(s) over which we shard tensor-parallel tensors.
        seq_axis_names: Axis name(s) over which we shard sequence-parallel tensors.
    """

    # pytype: disable=attribute-error
    def set_attn_partition_specs(attn_layer: MultiheadAttention.Config):
        # Shard weights.
        input_linear_cfg = attn_layer.input_linear
        if hasattr(input_linear_cfg, "input_linear"):
            input_linear_cfg = input_linear_cfg.input_linear
        input_linear_cfg.layer.param_partition_spec = (fsdp_axis_names, tp_axis_names, None)
        attn_layer.output_linear.param_partition_spec = (fsdp_axis_names, tp_axis_names, None)
        #attn_layer.output_linear.output_partition_spec = (batch_axis_names, seq_axis_names, None)

    def set_ffn_partition_specs(ff_layer: TransformerFeedForwardLayer.Config):
        # Shard weights.
        ff_layer.linear1.param_partition_spec = (fsdp_axis_names, tp_axis_names)
        ff_layer.linear2.param_partition_spec = (tp_axis_names, fsdp_axis_names)
        # Encourage the right activation sharding.
        ff_layer.linear1.output_partition_spec = (batch_axis_names, None, tp_axis_names)
        ff_layer.linear2.output_partition_spec = (batch_axis_names, seq_axis_names, None)

    if not isinstance(cfg, Sequence):
        cfg = [cfg]

    for layer_cfg in cfg:
        set_attn_partition_specs(layer_cfg.self_attention.attention)
        if layer_cfg.cross_attention is not None:
            set_attn_partition_specs(layer_cfg.cross_attention.attention)
        if isinstance(layer_cfg.feed_forward, TransformerFeedForwardLayer.Config):
            set_ffn_partition_specs(layer_cfg.feed_forward)


def collect_param_specs(layer):
    # If the layer has children, recurse into them and collect their specs.
    if hasattr(layer, '_children') and layer._children:
        return {name: collect_param_specs(child) for name, child in layer._children.items()}
    else:
        # Otherwise, return the parameter specs of the current layer.
        return layer._create_layer_parameter_specs()
