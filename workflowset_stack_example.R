library(tidymodels)
library(stacks)
library(dplyr)

# Sample Data (replace with your actual data)
data(mtcars)
mtcars_df <- as_tibble(mtcars)

# Split the Data
set.seed(123)
data_split <- initial_split(mtcars_df, prop = 0.75)
data_train <- training(data_split)
data_test <- testing(data_split)

#1. Verify numeric features
numeric_columns <- names(data_train)[sapply(data_train, is.numeric)]
print("Numeric columns as identified with `is_numeric`")
print(numeric_columns)

#2. Remove non-numeric columns such as strings and characters
data_train <- select(data_train, all_of(numeric_columns))
data_test <- select(data_test, all_of(numeric_columns))

# Set up the rest!

# Create Resamples for CV
data_folds <- vfold_cv(data_train, v = 5)

# Define Base Models (with tuning)
lr_model <- linear_reg(penalty = tune(), mixture = tune()) |>
  set_engine("glmnet")

rf_model <- rand_forest(mtry = tune(), min_n = tune(), trees = 500) |>
  set_engine("ranger") |>
  set_mode("regression")

# *Important: Set up preprocessing using Recipes
lr_recipe <-
  recipe(mpg ~ ., data = data_train) |>
  step_normalize(all_numeric_predictors()) #Scale our numeric data

rf_recipe <-
  recipe(mpg ~ ., data = data_train) |>
  step_normalize(all_numeric_predictors()) #Scale our numeric data

# Set up workflows: Combines recipe and model in one
#Create workflow
# Set up workflows: Combines recipe and model in one
#Workflow functions MUST be set up again outside of workflow_set.
lr_workflow <-
  workflow() |>
  add_recipe(lr_recipe) |>
  add_model(lr_model)

#Create workflow
rf_workflow <-
  workflow() |>
  add_recipe(rf_recipe) |>
  add_model(rf_model)


#Tuning and grid
lr_grid <- grid_regular(penalty() , mixture(), levels = 3 )

rf_grid <- grid_regular(mtry(range = c(1 , 5)), min_n () , levels = 2)

#Tune grids
lr_tuned <- tune_grid(
  lr_workflow,
  resamples = data_folds,
  #Metrics = metric_set(),
  grid = lr_grid
)

#Tune grids
rf_tuned <- tune_grid(
  rf_workflow,
  resamples = data_folds,
  #Metrics = metric_set(rsq , rmse),
  grid = rf_grid
)
ctrl <- control_resamples(save_pred = TRUE, save_workflow = TRUE,verbose=TRUE)
#Combine workflow_set function. Use list to generate workflows_set
workflows_models <- workflow_set(

  preproc = list(LinearModel = lr_recipe , RandomForest = rf_recipe),
  models = list(LinearModel = lr_model, RandomForest = rf_model),
  cross = FALSE
)
# 6. Custom Tuning Function (Crucial Step)
# This function decides which grid to use for each workflow
tune_with_appropriate_grid <- function(workflow, resamples, metrics, ...) {
  model_id <- workflow$id
  grid <- switch(model_id,
                 xgboost = xgb_lgbm_grid,
                 lightgbm = xgb_lgbm_grid,
                 glmnet = glmnet_grid,
                 ranger = ranger_grid,
                 knn = knn_grid,
                 lm = lm_grid,
                 NULL # Default: No grid
  )

  if (is.null(grid)) {
    #For untuned mode
    fit_resamples(
      workflow,
      resamples = resamples,
      metrics = metrics,
      control = control_resamples(save_pred = FALSE, save_workflow = FALSE)
    )
  } else {
    tune_grid(
      workflow,
      resamples = resamples,
      grid = grid,
      metrics = metrics,
      control = control_grid(save_pred = FALSE, save_workflow = FALSE)
    )
  }
}

# 7.  Tune the Models using the Custom Function
ctrl <- control_resamples(save_pred = TRUE, save_workflow = TRUE,verbose=TRUE)
workflows_tuned <- workflows_models |>
  workflow_map(
    fn = "tune_grid",
    resamples = data_folds,
    control=ctrl,
    metrics = metric_set(rmse, rsq)  # Pass metrics
  )
workflows_tuned |>collect_metrics()

# 8. Continue with Stacking (Blending and Meta-learner) - As Before

stacks_mod <-stacks() |>
  add_candidates(workflows_tuned)  # Use  workflows_tuned Here!
collect_parameters(stacks_mod, "LinearModel_LinearModel")
# Blend
blended_mod <- stacks_mod |>
  blend_predictions(resamples = data_folds)
