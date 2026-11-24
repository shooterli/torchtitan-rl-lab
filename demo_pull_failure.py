"""CPU example of forwarding a failed weight pull to a waiting trainer."""

import asyncio


async def consume_batch(queue: asyncio.Queue) -> None:
    batch = await queue.get()
    if isinstance(batch, Exception):
        raise batch


async def failing_pull(queue: asyncio.Queue, forward_error: bool) -> None:
    await asyncio.sleep(0)
    try:
        raise RuntimeError("synthetic weight pull failure")
    except RuntimeError as error:
        if forward_error:
            queue.put_nowait(error)
        raise


async def demonstrate(forward_error: bool) -> None:
    queue = asyncio.Queue(maxsize=1)
    trainer = asyncio.create_task(consume_batch(queue))
    pull = asyncio.create_task(failing_pull(queue, forward_error))
    try:
        try:
            await pull
        except RuntimeError as error:
            assert str(error) == "synthetic weight pull failure"
        if forward_error:
            try:
                await asyncio.wait_for(trainer, timeout=1)
            except RuntimeError as error:
                assert str(error) == "synthetic weight pull failure"
                print("Forward error: trainer raised the original pull failure.")
            else:
                raise AssertionError("trainer did not raise the pull failure")
        else:
            assert not trainer.done()
            print("Isolated error: pull failed, trainer is still waiting for a batch.")
    finally:
        for task in (trainer, pull):
            task.cancel()
        await asyncio.gather(trainer, pull, return_exceptions=True)


async def main() -> None:
    await demonstrate(forward_error=False)
    await demonstrate(forward_error=True)


if __name__ == "__main__":
    asyncio.run(main())
