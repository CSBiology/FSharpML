namespace FSharpML

open Microsoft.ML
open Microsoft.ML.Data
open Microsoft.ML.Core.Data
open Microsoft.Data.DataView

/// Module to handel transformer
module Transformer =


    let downcastTransformer (transformer : ITransformer) =
        match transformer with
        | :? IPredictionTransformer<IPredictor> as p -> p
        | _ -> failwith "The transformer has to be an instance of IPredictionTransformer<IPredictor>."

 
    /// 
    let append (source1 : ITransformer) (source2 : ITransformer)  = 
        (source2 |> downcastTransformer).Append(source1) 
     
    ///
    let createTransformerChainOf (estimators : ITransformer seq) =
        estimators
        |> Seq.fold (fun acc e -> append e acc) (TransformerChain())



        