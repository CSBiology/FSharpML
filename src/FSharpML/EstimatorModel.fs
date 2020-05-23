namespace FSharpML.EstimatorModel

open Microsoft.ML
open Microsoft.ML.Data



/// Module to build an estimator model that represents the estimator procedure  
[<AutoOpen>]
module EstimatorModel =
    open FSharpML
    open FSharpML.TransformerModel

    /// Represents the estimator procedure
    type EstimatorModel<'a when 'a :> ITransformer and 'a :not struct> = {
        EstimatorChain : EstimatorChain<'a>
        Context: MLContext
        }
    
    /// Returns the MLContext from an EstimatorModel 
    let getContext (estimatorModel:EstimatorModel<_>) = estimatorModel.Context

    /// Returns the EstimatorChain from an EstimatorModel
    let getEstimatorChain (estimatorModel:EstimatorModel<_>) = estimatorModel.EstimatorChain
 
    /// Create an empty EstimatorModel and binds it to a MlContext
    let create (mlContext : MLContext) = 
        {EstimatorChain=EstimatorChain();Context=mlContext}
        
    /// Appends a CacheCheckpoint to chache the estimator
    let appendCacheCheckpoint (estimatorModel:EstimatorModel<_>) =
        let chain = 
            estimatorModel.EstimatorChain.AppendCacheCheckpoint estimatorModel.Context
        {EstimatorChain=chain;Context=estimatorModel.Context}

    ///
    let appendBy (transforming:MLContext -> #IEstimator<_>) (estimatorModel:EstimatorModel<_>) =
        let estimator = transforming estimatorModel.Context
        let chain =
            estimatorModel.EstimatorChain
            |> Estimator.append estimator
        {EstimatorChain=chain;Context=estimatorModel.Context}


    /// Transforms a columns according to the given function from TransformsCatalog
    let transformBy (transformsCatalog:TransformsCatalog -> #IEstimator<_>) (estimatorModel:EstimatorModel<_>) =
        estimatorModel
        |> appendBy (fun mlc -> transformsCatalog mlc.Transforms)  


    ///
    let fit (data:IDataView) (estimatorModel:EstimatorModel<_>) = 
        let transformer = estimatorModel.EstimatorChain.Fit data
        {TransformerChain=transformer;Context=estimatorModel.Context}



