# Logging


#### How to to see debug logs?

The most straightforward way is to enable debug logs for the entire project via hydra.

```bash
hydra.verbose=true
```

However, this will print a lot of logs from other libraries such as `urllib3` and `openai`.

To enable debug logs only for the project, you need to do the following:

```bash
hydra.verbose=[src]
```

This will enable debug logs for the `src` modul, i.e. the entire project.

Check [here](https://hydra.cc/docs/tutorials/basic/running_your_app/logging/) for more details.

#### How to enable debug logs without hydra?

In the main script, add the following lines:

```python
import logging

logging.getLogger('src').setLevel(logging.DEBUG)
```
This will enable debug logs for the `src` module.
You can surely enable debug logs for other modules or submodules.


#### Why there is replication of logs commands?

```python
self.log(f"test/precision", running_p, rank_zero_only=True)
self.log("test/recall", running_r, rank_zero_only=True)
self.log("test/f1", running_f1, rank_zero_only=True)

log.info(f"test/precision: {running_p}")
log.info(f"test/recall: {running_r}")
log.info(f"test/f1: {running_f1}")
```

The first three lines are for logging to pytorch lightning, while the last three lines are for logging to the console.
The pytorch lightning logger takes care of writing the logs to local file and also to remote loggers such as wandb.

#### I see logs from libraries such as openai and urllib3, how to disable them?

```bash
import logging
logging.getLogger('openai').setLevel(logging.INFO)
logging.getLogger('urllib3').setLevel(logging.INFO)
```