## Scenario

User tracks **_signals_** related to a list of entities of interest. 

Views:
(1) the top-level view is a set of feeds that I'm monitoring
(2) the anomaly view shows the cards for the times when a feed trended
    - anomaly cards are bounded by start and end dates -- this is the "window of interest"
(3) implicitly there is an "all events" view which is every item that passed the feed's filters

### Running the demo

(3) run the user interface
```
make run
```