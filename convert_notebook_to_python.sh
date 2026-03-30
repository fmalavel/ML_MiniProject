# create a list containing the names of the notebooks to convert
# notebooks=(
#     "Training_MLP.ipynb"
#     "Training_CNN+LSTM.ipynb"
#     "Training_CNN.ipynb"
#     "Training_SVM.ipynb"
#     "Training_UNet.ipynb"
#     "Training_convLSTM_met_only_train_from_daily.ipynb"
#     "Training_convLSTM_met_only_train_from_hourly.ipynb"
#     "Training_convLSTM_with_rasterised_ozone_train_from_daily.ipynb"
#     "Training_convLSTM_with_rasterised_ozone_train_from_hourly.ipynb"
# )

notebooks=(
    "Training_generic.ipynb"
)

for notebook in "${notebooks[@]}"; do
    jupyter nbconvert --to python --TemplateExporter.exclude_markdown=True "$notebook"
done
