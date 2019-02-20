namespace FSharpML

open Microsoft.ML
open Microsoft.ML.Data
open Microsoft.ML.Core.Data
open Microsoft.Data.DataView
open TransformerModel

module EstimatorModel =


    type EstimatorModel<'a when 'a :> ITransformer and 'a :not struct> = {
        EstimatorChain : EstimatorChain<'a>
        Context: MLContext
        }
    
    let getContext (estimatorModel:EstimatorModel<_>) = estimatorModel.Context

    let getEstimatorChain (estimatorModel:EstimatorModel<_>) = estimatorModel.EstimatorChain

    let downcastEstimator (estimator : IEstimator<'a>) =
        match estimator with
        | :? IEstimator<ITransformer> as p -> p
        | _ -> failwith "The estimator has to be an instance of IEstimator<ITransformer>."
 
    /// Create an empty EstimatorChain
    let create (mlContext : MLContext) = 
        {EstimatorChain=EstimatorChain();Context=mlContext}
        

    /// Create a new EstimatorChain by appending another Estimator
    let append (source1 : IEstimator<'a>) (source2 : IEstimator<'b>)  = 
        (source2 |> downcastEstimator).Append(source1) 
     
    ///
    let createEstimatorModelOf (estimators : IEstimator<'a> seq) =
        estimators
        |> Seq.fold (fun acc e -> append e acc) (EstimatorChain())

    ///
    let appendCacheCheckpoint (estimatorModel:EstimatorModel<_>) =
        let chain = 
            estimatorModel.EstimatorChain.AppendCacheCheckpoint estimatorModel.Context
        {EstimatorChain=chain;Context=estimatorModel.Context}

    ///
    let map (mapping:MLContext -> #IEstimator<_>) (estimatorModel:EstimatorModel<_>) =
        let estimator = mapping estimatorModel.Context
        let chain =
            estimatorModel.EstimatorChain
            |> append estimator
        {EstimatorChain=chain;Context=estimatorModel.Context}

    ///
    let fit (data:IDataView) (estimatorModel:EstimatorModel<_>) = 
        let transformer = estimatorModel.EstimatorChain.Fit data
        {TransformerChain=transformer;Context=estimatorModel.Context}