# Logging


#### How to to see debug logs?

You need to set the logging level to `logging.DEBUG`



In the main script, add the following lines:

```python
import logging

logging.getLogger('src.models').setLevel(logging.DEBUG)
```
This will enable debug logs for the `src.models` module.
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