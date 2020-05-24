namespace FSharpML.EstimatorModel

open Microsoft.ML
open Microsoft.ML.Data


/// Module to handel data transformation
module Transforms =

    //open Microsoft.ML.Transforms.Normalizers

    /// Concatenate columns together
    let concatenate outputColumnName inputColumnNames (estimatorModel:EstimatorModel<_>) =
        estimatorModel
        |> appendBy (fun mlc -> mlc.Transforms.Concatenate(outputColumnName,inputColumnNames))
    
    /// Copy column from source to target
    let copyColumn sourceColumnName targetColumnName (estimatorModel:EstimatorModel<_>) =
        estimatorModel
        |> appendBy (fun mlc -> mlc.Transforms.CopyColumns(sourceColumnName,targetColumnName))  

    /// Transforms a columns according to the given function from TransformsCatalog
    let map (transformsCatalog:TransformsCatalog -> #IEstimator<_>) (estimatorModel:EstimatorModel<_>) =
        estimatorModel
        |> appendBy (fun mlc -> transformsCatalog mlc.Transforms)  


    ///// Copy columns from source to target
    //let copyColumns (columnNames:seq<string*string>) (estimatorModel:EstimatorModel<_>) =
    //    estimatorModel
    //    |> appendBy (fun mlc -> mlc.Transforms.CopyColumns(columnNames |> Seq.map (fun (a,b) -> struct (a,b)) |> Seq.toArray ))    
        

    ///// Normalize the columns according to spezified mode
    //let normalizeWith mode outputColumnNames inputColumnNames (estimatorModel:EstimatorModel<_>) =
    //    estimatorModel
    //    |> appendBy (fun mlc -> mlc.Transforms.N Normalize(outputColumnNames,inputColumnNames,mode))
   
    ///// Normalize the columns MeanVariance
    //let normalizeMeanVariance outputColumnNames inputColumnNames (estimatorModel:EstimatorModel<_>) =
    //    estimatorModel
    //    |> appendBy (fun mlc -> mlc.Transforms.Normalize(outputColumnNames,inputColumnNames,NormalizingEstimator.NormalizerMode.MeanVariance ))  
            
    ///// Normalize the columns Binning
    //let normalizeBinning outputColumnNames inputColumnNames (estimatorModel:EstimatorModel<_>) =
    //    estimatorModel
    //    |> appendBy (fun mlc -> mlc.Transforms.Normalize(outputColumnNames,inputColumnNames,NormalizingEstimator.NormalizerMode.Binning ))

    ///// Normalize the columns LogMeanVariance
    //let normalizeLogMeanVariancee outputColumnNames inputColumnNames (estimatorModel:EstimatorModel<_>) =
    //    estimatorModel
    //    |> appendBy (fun mlc -> mlc.Transforms.Normalize(outputColumnNames,inputColumnNames,NormalizingEstimator.NormalizerMode.LogMeanVariance ))

    ///// Normalize the columns MinMax
    //let normalizeMinMax outputColumnNames inputColumnNames (estimatorModel:EstimatorModel<_>) =
    //    estimatorModel
    //    |> appendBy (fun mlc -> mlc.Transforms.Normalize(outputColumnNames,inputColumnNames,NormalizingEstimator.NormalizerMode.MinMax ))

    ///// Normalize the columns SupervisedBinning
    //let normalizeSupervisedBinning mode outputColumnNames inputColumnNames (estimatorModel:EstimatorModel<_>) =
    //    estimatorModel
    //    |> appendBy (fun mlc -> mlc.Transforms.Normalize(outputColumnNames,inputColumnNames,NormalizingEstimator.NormalizerMode.SupervisedBinning ))


    //module Categorical =

    //    /// Converts a text column into one-hot encoded vector 
    //    let oneHotEncoding outputColumnName inputColumnName (estimatorModel:EstimatorModel<_>) =
    //        estimatorModel
    //        |> appendBy (fun mlc -> mlc.Transforms.Categorical.OneHotEncoding(outputColumnName,inputColumnName) )

    //    /// Converts a text column into one-hot encoded vector 
    //    let oneHotEncodingBag outputColumnName inputColumnName (estimatorModel:EstimatorModel<_>) =
    //        estimatorModel
    //        |> appendBy (fun mlc ->
    //            mlc.Transforms.Categorical.OneHotEncoding(outputColumnName,inputColumnName,Transforms.Categorical.OneHotEncodingTransformer.OutputKind.Bag))

    //    /// Converts a text column into one-hot encoded vector 
    //    let oneHotEncodingBin outputColumnName inputColumnName (estimatorModel:EstimatorModel<_>) =
    //        estimatorModel
    //        |> appendBy (fun mlc -> 
    //                mlc.Transforms.Categorical.OneHotEncoding(outputColumnName,inputColumnName,Transforms.Categorical.OneHotEncodingTransformer.OutputKind.Bin))
        
    //    /// Converts a text column into one-hot encoded vector 
    //    let oneHotEncodingInd outputColumnName inputColumnName (estimatorModel:EstimatorModel<_>) =
    //        estimatorModel
    //        |> appendBy (fun mlc -> 
    //            mlc.Transforms.Categorical.OneHotEncoding(outputColumnName,inputColumnName,Transforms.Categorical.OneHotEncodingTransformer.OutputKind.Ind))

    //    /// Converts a text column into one-hot encoded vector 
    //    let oneHotEncodingKey outputColumnName inputColumnName (estimatorModel:EstimatorModel<_>) =
    //        estimatorModel
    //        |> appendBy (fun mlc -> 
    //            mlc.Transforms.Categorical.OneHotEncoding(outputColumnName,inputColumnName,Transforms.Categorical.OneHotEncodingTransformer.OutputKind.Key))



///// Module to handel data transformation
//module Transforms1 =

//    /// Concatenate columns together
//    let concatenate outputColumnName inputColumnNames (mlc:MLContext) =
//        mlc.Transforms.Concatenate(outputColumnName,inputColumnNames)
    
//    /// Copy column from source to target
//    let copyColumn sourceColumnName targetColumnName (mlc:MLContext) =
//        mlc.Transforms.CopyColumns(sourceColumnName,targetColumnName)    

//    /// Copy columns from source to target
//    let copyColumns (columnNames:seq<string*string>) (mlc:MLContext) =
//        mlc.Transforms.CopyColumns(columnNames |> Seq.map (fun (a,b) -> struct (a,b)) |> Seq.toArray )        
        

//    /// Normalize the columns according to spezified mode
//    let normalize mode outputColumnNames inputColumnNames (mlc:MLContext) =
//        mlc.Transforms.Normalize(outputColumnNames,inputColumnNames,mode)
    
//    module Categorical =

//        /// Converts a text column into one-hot encoded vector 
//        let oneHotEncoding outputColumnName inputColumnName (mlc:MLContext) =
//            mlc.Transforms.Categorical.OneHotEncoding(outputColumnName,inputColumnName)

//        /// Converts a text column into one-hot encoded vector 
//        let oneHotEncodingBag outputColumnName inputColumnName (mlc:MLContext) =
//            mlc.Transforms.Categorical.OneHotEncoding(outputColumnName,inputColumnName,Transforms.Categorical.OneHotEncodingTransformer.OutputKind.Bag)

//        /// Converts a text column into one-hot encoded vector 
//        let oneHotEncodingBin outputColumnName inputColumnName (mlc:MLContext) =
//            mlc.Transforms.Categorical.OneHotEncoding(outputColumnName,inputColumnName,Transforms.Categorical.OneHotEncodingTransformer.OutputKind.Bin)
        
//        /// Converts a text column into one-hot encoded vector 
//        let oneHotEncodingInd outputColumnName inputColumnName (mlc:MLContext) =
//            mlc.Transforms.Categorical.OneHotEncoding(outputColumnName,inputColumnName,Transforms.Categorical.OneHotEncodingTransformer.OutputKind.Ind)

//        /// Converts a text column into one-hot encoded vector 
//        let oneHotEncodingKey outputColumnName inputColumnName (mlc:MLContext) =
//            mlc.Transforms.Categorical.OneHotEncoding(outputColumnName,inputColumnName,Transforms.Categorical.OneHotEncodingTransformer.OutputKind.Key)



