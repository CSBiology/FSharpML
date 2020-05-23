namespace FSharpML

open Microsoft.ML
open Microsoft.ML.Data


/// Module to handel estimators 
module Estimator =

    let downcastEstimator (estimator : IEstimator<'a>) =
        match estimator with
        | :? IEstimator<ITransformer> as p -> p
        | _ -> failwith "The estimator has to be an instance of IEstimator<ITransformer>."
 
    /// 
    let append (source1 : IEstimator<'a>) (source2 : IEstimator<'b>)  = 
        (source2|> downcastEstimator).Append(source1) 
     
    ///
    let createEstimatorChainOf (estimators : IEstimator<'a> seq) =
        estimators
        |> Seq.fold (fun acc e -> append e acc) (EstimatorChain())

    ///
    let appendCacheCheckpoint (mlContext : MLContext) (pipeline: IEstimator<'a>) =
        pipeline.AppendCacheCheckpoint mlContext
        |> downcastEstimator

        