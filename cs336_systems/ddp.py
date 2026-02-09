"""Naive distributed data parallel training implementation."""

import os
from copy import deepcopy

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _setup(rank: int, world_size: int, backend: str = "gloo") -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def _data(
    global_batchsize: int,
    shape: tuple[int, ...] = (3, 64),
    seed: int = 42,
) -> torch.Tensor:
    # ignore device, move data to rank_i or cpu in ddp training loop
    # generate in cpu to keep multi device stable,
    # move to gpu in ddp training loop when necessary
    return torch.randn(
        global_batchsize,
        *shape,
        generator=torch.Generator().manual_seed(seed),
    )


def _rank_parallel_train(
    rank: int,
    world_size: int,
    backend: str,
    global_batchsize: int,
    iters: int,
) -> None:
    if backend == "nccl":
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
    else:
        device = torch.device("cpu")
    _setup(rank, world_size, backend)
    microbatch_size = global_batchsize // world_size

    seed = 42
    global_data = _data(global_batchsize=global_batchsize, seed=seed).to(device)
    micro_data = global_data[rank * microbatch_size : (rank + 1) * microbatch_size]

    dim = micro_data.shape[-1]
    base_model = torch.nn.Linear(dim, dim).to(device)

    # different model init for rank's model,
    # using broadcast to sync rank0's in actul training
    torch.manual_seed(seed + rank)
    ddp_model = deepcopy(base_model)
    for param in ddp_model.parameters():
        dist.broadcast(param.data, src=0)

    base_optimizer = torch.optim.SGD(base_model.parameters(), lr=0.01)
    ddp_optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.01)

    for _ in range(iters):
        base_loss = base_model(global_data).mean()
        base_loss.backward()
        base_optimizer.step()
        base_optimizer.zero_grad()

        ddp_loss = ddp_model(micro_data).mean()
        ddp_loss.backward()
        # loss mean + all reduce mean equal to global batch size mean
        for param in ddp_model.parameters():
            dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
        ddp_optimizer.step()
        ddp_optimizer.zero_grad()

        if rank == 0:
            for param_base, param_ddp in zip(
                base_model.parameters(), ddp_model.parameters(), strict=True
            ):
                # if not raise AssertionError, the two models are close enough after one step of training
                assert torch.allclose(param_base.data, param_ddp.data, atol=1e-6)

    dist.destroy_process_group()


def naive_ddp(batch_size: int = 32, n_procs: int = 8, backend: str = "gloo") -> None:
    """Verify naive DDP training matches single-process training."""
    assert batch_size % n_procs == 0
    iters = 10
    mp.spawn(
        _rank_parallel_train,
        args=(n_procs, backend, batch_size, iters),
        nprocs=n_procs,
        join=True,
    )


if __name__ == "__main__":
    naive_ddp(batch_size=32, n_procs=8, backend="gloo")
