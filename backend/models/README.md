# Models

Place the production-ready inference artifacts for the chosen model here.

Examples:

- exported checkpoint files (`.pt`, `.pth`)
- preprocessing metadata
- label maps

Once the final model is selected, wire it into
`backend/app/model_service.py`.

Current Render default:

- `101_weighted_cnn_preprocessed_ensemble_densenet121_best.pt`
- `101_weighted_cnn_preprocessed_ensemble_efficientnet_b0_best.pt`
- `101_weighted_cnn_preprocessed_ensemble_resnet18_best.pt`
