namespace FSharpML

open Microsoft.ML
open Microsoft.ML.Data
open Microsoft.ML.Core.Data
open Microsoft.Data.DataView

/// Module to handel estimators 
module Estimator =

    let downcastEstimator (estimator : IEstimator<'a>) =
        match estimator with
        | :? IEstimator<ITransformer> as p -> p
        | _ -> failwith "The estimator has to be an instance of IEstimator<ITransformer>."
 
    /// 
    let append (source1 : IEstimator<'a>) (source2 : IEstimator<'b>)  = 
        (source2 |> downcastEstimator).Append(source1) 
     
    ///
    let createEstimatorModelOf (estimators : IEstimator<'a> seq) =
        estimators
        |> Seq.fold (fun acc e -> append e acc) (EstimatorChain())



        