# Cell Migration


## Notes

### 27.10.24 -- Bayesflow
- Each summary net and amortizer should come with more documentation of the type of data it expects
- Without this it is very hard to write a configurator function for a summary network

### 7.11.24 -- Bayesflow
- Is there a good reason for the trainer to take a configurator?
- Trainer should not take a configurator function, the user should be able to config the data and then just give it to the trainer, at least for offline
- There needs to be a lot more documentation on the website about what SBC ECDF is

### 27.02.25
- Maybe online training is worth looking at to end relying on presimulation
- Morpheus does not have a Python API, that can directly generate data for us