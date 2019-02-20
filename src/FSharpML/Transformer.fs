namespace FSharpML

open Microsoft.ML
open Microsoft.ML.Core.Data


module Transformer =

    let downcastTransformer (transformer : ITransformer) =
        match transformer with
        | :? IPredictionTransformer<IPredictor> as p -> p
        | _ -> failwith "The transformer has to be an instance of IPredictionTransformer<IPredictor>."

