# Logging


#### How to change the logging level of a logger?

In each script, the logging level of each logger can be changed by the following:

```python
log = utils.get_only_rank_zero_logger(__name__, stdout=True)
import logging
log.setLevel(logging.INFO) # or any other level
```

By default, the logging level is set to `logging.INFO` in `utils.get_only_rank_zero_logger`.



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
