# Real-Time Bus Headway Analysis and Anomaly Detection for Madrid & London

This repository contains the code and data used in the publication:

> A. Jarabo-Peñas, P. J. Zufiria and C. García-Mauriño, "Bus Headways Analysis for Anomaly Detection," in *IEEE Transactions on Intelligent Transportation Systems*, vol. 23, no. 10, pp. 18975-18988, Oct. 2022, doi: 10.1109/TITS.2022.3155180.

**Keywords:** Estimation; Anomaly detection; Real-time systems; Indexes; Global Positioning System; Data models; Urban areas; Bus; headway; monitoring; anomaly detection.

**Abstract:** This project develops a data processing system to analyze and model bus headways (time intervals between consecutive vehicles) in Madrid and London, using real-time data from their respective public transport APIs. The goal is to detect operational anomalies. By statistically modeling the evolution of headways, we create an anomaly detection scheme and a Quality of Service (QoS) index. This system enables online anomaly detection, improving the efficiency of bus transportation services.

## Project Overview

This research explores the use of real-time bus data to monitor and detect anomalies in bus headways. The project focuses on:

* **Data Collection and Processing:** Gathering and cleaning real-time bus data from Madrid and London.
* **Headway Analysis:** Modeling the statistical behavior of bus headways.
* **Anomaly Detection:** Implementing an unsupervised anomaly detection scheme.
* **Dashboard Visualization:** Providing a dashboard for real-time monitoring and analysis.

## Data

### Download Pre-Processed Data

The processed data files are available for download:

* **Link:** [https://mega.nz/folder/QRIGnQRZ#7fJVQcapLkSp7jGGz0WZeQ](https://mega.nz/folder/QRIGnQRZ#7fJVQcapLkSp7jGGz0WZeQ)
* **Files:**
    * `buses_data_cleaned.csv`: Cleaned raw data.
    * `arrival_times.csv`: Arrival times of each bus at each stop.
    * `time_bt_stops.csv`: Travel times between stops.
    * `headways.csv`: Headways between buses.
* **Location:** Place these files in the `/Data/Processed/` directory.

### Process Raw Data (Optional)

1.  Download the raw data from the link above or collect it using the script `/Scripts/CollectData/retrieve_data.py`. Follow the real-time data collection steps below.
2.  Run the processing pipeline using the scripts in `/Scripts/ProcessData/`. Execute `queue.sh` to process the raw data located in `/Data/Raw/buses_data.csv`.

### Real-Time Data Collection

1.  **API Account:** Create an account at [https://mobilitylabs.emtmadrid.es/](https://mobilitylabs.emtmadrid.es/).
2.  **API Credentials:** Create a file named `api_credentials.py` in the root directory and add your API credentials.
3.  **Run Script:** Execute `/Scripts/CollectData/retrieve_data.py` to start collecting real-time bus data.

## Dashboard

### Setup

1.  **Install Dependencies:** Install the required Python packages using:

    ```bash
    pip install -r requirements.txt
    ```

2.  **Run Server:** Start the dashboard server using:

    ```bash
    bash run_server.sh
    ```

3.  **Access Dashboard:** Open your web browser and navigate to `0.0.0.0:8050`.

## Publication Context

This project aims to address the challenge of maintaining regular bus headways, crucial for efficient public transport. By developing an automatic anomaly detection scheme, this research contributes to improving the Quality of Service (QoS) in urban bus networks.

The paper details the methodology used to clean data, estimate headways, and develop the anomaly detection algorithm. It also outlines the statistical characterization of headways and the implementation of the unsupervised anomaly detection scheme.

For detailed information on the data processing, headway estimation, and statistical modeling, refer to the original publication.

## Notation

A detailed description of the notation used in the research can be found within the publication itself.
