import streamlit as st
import pandas as pd
from prophet import Prophet
from datetime import datetime
import calendar
from prophet.plot import plot_components_plotly
import plotly.graph_objs as go


def forecast_data1(data, forecast_date):
  # Initialize Prophet model
  m = Prophet()

  # Prepare data for Prophet
  data.columns = ['ds', 'y']

  # Fit the model
  model = m.fit(data)

  # Calculate the number of days for forecasting
  selected_date = pd.to_datetime(
      forecast_date, format="%d-%m-%Y")  # Specify the correct date format
  current_date = pd.to_datetime(datetime.today().strftime('%Y-%m-%d'))
  days_difference = (selected_date - current_date).days

  # Make future dataframe
  future = m.make_future_dataframe(periods=days_difference, freq='D')

  # Perform forecasting
  forecast = m.predict(future)

  # Plot the forecast
  st.subheader("Forecast Plot:")
  st.write(plot_components_plotly(m, forecast))


def forecast_data(data, forecast_date):
  # Initialize Prophet model
  m = Prophet()

  # Prepare data for Prophet
  data.columns = ['ds', 'y']

  # Fit the model
  model = m.fit(data)

  # Calculate the number of days for forecasting
  selected_date = pd.to_datetime(
      forecast_date, format="%d-%m-%Y")  # Specify the correct date format
  current_date = pd.to_datetime(datetime.today().strftime('%Y-%m-%d'))
  days_difference = (selected_date - current_date).days

  # Make future dataframe
  future = m.make_future_dataframe(periods=days_difference, freq='D')

  # Perform forecasting
  forecast = m.predict(future)

  # Plot the forecast
  # st.subheader("Forecast Plot:")
  # st.write(plot_components_plotly(m, forecast))

  return forecast


def main():
  st.title("Youtube Predictifier WebApp")

  # Upload data file
  st.sidebar.title("Upload Data")
  data_file = st.sidebar.file_uploader("Upload your data file (CSV format):",
                                       type=['csv'])

  if data_file is not None:
    # Load data
    data = pd.read_csv(data_file)
    data['Date'] = pd.to_datetime(
        data['Date'], format="%d-%m-%Y")  # Specify the correct date format

    # Display uploaded data
    st.subheader("Your Uploaded Data:")
    st.dataframe(data.head(), use_container_width=True)
    st.divider()

    # Select forecast date
    forecast_date = st.sidebar.date_input("Select Forecast Date:")

    if st.sidebar.button("Forecast"):

      # Forecast Subscribers
      st.subheader(f"Forecast for Subscribers on {forecast_date}:")
      forecast_subscribers = forecast_data(data[['Date', 'Subscribers']],
                                           forecast_date)

      st.write("Total Subscribers : ", int(abs(forecast_subscribers['yhat'].sum())))

      # Find the top 6 landmarks for Subscribers
      top_landmarks_subscribers = forecast_subscribers.nlargest(
          6, 'yhat')[['ds', 'yhat']]
      top_landmarks_subscribers['ds'] = top_landmarks_subscribers[
          'ds'].dt.strftime("%B %d, %Y")
      top_landmarks_subscribers.columns = ['Date', 'Landmarks']

      # Display top 6 landmarks for Subscribers
      st.subheader("Subscriber Landmarks:")
      st.dataframe(top_landmarks_subscribers, use_container_width=True)

      # plotting

      forecast_data1(data[['Date', 'Subscribers']], forecast_date)

      st.divider()

      # Forecast Views
      st.subheader(f"Forecast for Views on {forecast_date}:")
      forecast_views = forecast_data(data[['Date', 'Views']], forecast_date)

      st.write("Total Views :", int(abs(forecast_views['yhat'].sum())))

      # Find the top 6 landmarks for Views
      top_landmarks_views = forecast_views.nlargest(6, 'yhat')[['ds', 'yhat']]
      top_landmarks_views['ds'] = top_landmarks_views['ds'].dt.strftime(
          "%B %d, %Y")
      top_landmarks_views.columns = ['Date', 'Landmarks']

      # Display top 6 landmarks for Views
      st.subheader("Views Landmarks:")
      st.dataframe(top_landmarks_views, use_container_width=True)

      # plotting
      forecast_data1(data[['Date', 'Views']], forecast_date)

      st.divider()

      # Forecast Revenue
      st.header(f"Forecast for Revenue on {forecast_date}:")
      forecast_revenue = forecast_data(data[['Date', 'Revenue']],
                                       forecast_date)

      st.write("Total Revenue :", int(abs(forecast_revenue['yhat'].sum())))

      # Find the top 6 landmarks for Revenue
      top_landmarks_revenue = forecast_revenue.nlargest(6,
                                                        'yhat')[['ds', 'yhat']]
      top_landmarks_revenue['ds'] = top_landmarks_revenue['ds'].dt.strftime(
          "%B %d, %Y")
      top_landmarks_revenue.columns = ['Date', 'Landmarks']

      # Display top 6 landmarks for Revenue
      st.subheader("Revenue Landmarks:")
      st.dataframe(top_landmarks_revenue, use_container_width=True)

      # plotting
      forecast_data1(data[['Date', 'Revenue']], forecast_date)


if __name__ == "__main__":
  main()
