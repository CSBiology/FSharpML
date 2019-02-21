namespace FSharpML

open Microsoft.ML
open Microsoft.ML.Core.Data


module TransformerModel =
    open Microsoft.ML.Data

    type TransformerModel<'a when 'a :> ITransformer and 'a :not struct> = {
        TransformerChain : TransformerChain<'a>
        Context: MLContext
        }

    ///
    let getContext (transformerModel:TransformerModel<_>) = transformerModel.Context

    ///
    let getTransformerChain (transformerModel:TransformerModel<_>) = transformerModel.TransformerChain


    ///
    let transform data (transformerModel:TransformerModel<_>) =
        transformerModel.TransformerChain.Transform data