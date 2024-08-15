##COPYRIGHT STEFAN L. BUND, copyright available at https://github.com/stefanbund/radidisco. No use is authorized and no sharing license is granted.
#!/bin/bash
while true; do
    # Start the application in the background
    python3 Predictor.py &  #Predictor-CAPSTONE.py
    APP_PID=$!

    # Run the application for 3 minutes (180 seconds)
    sleep 180

    # Pause the application
    kill -STOP $APP_PID

    # Wait for a desired pause duration
    sleep 100 #<pause_duration>

    # Restart the application
    kill -CONT $APP_PID

    # Run the application for another 3 minutes
    sleep 180

    # Optionally, stop the application after the time has elapsed
    kill $APP_PID
done