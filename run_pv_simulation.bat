@echo off
REM Call the activate script using its full path and specify the environment name.
call "C:\Users\Lospsy\Anaconda3\Scripts\activate.bat" Thesis

REM Now call Python using its full path in the Thesis environment.
"C:\Users\Lospsy\Anaconda3\python.exe" "C:/Users/Lospsy/Desktop/Thesis/pv_simulation_V8.py" ^
  --data_file "C:/Users/Lospsy/Desktop/Thesis/DATA_kkav.csv" ^
  --output_dir "C:/Users/Lospsy/Desktop/Thesis/Results" ^
  --config_file "C:/Users/Lospsy/Desktop/Thesis/config.yaml" ^
  --latitude 37.98983 ^
  --longitude 23.74328

pause
