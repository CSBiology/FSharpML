namespace FSharpML

open Microsoft.ML
open Microsoft.ML.Data
open Microsoft.ML.Core.Data
open Microsoft.Data.DataView


/// Module to handel data transformation
module Transforms =

    /// Concatenate columns together
    let concatenate outputColumnName inputColumnNames (mlc:MLContext) =
        mlc.Transforms.Concatenate(outputColumnName,inputColumnNames)
    
    /// Copy column from source to target
    let copyColumn sourceColumnName targetColumnName (mlc:MLContext) =
        mlc.Transforms.CopyColumns(sourceColumnName,targetColumnName)        

    /// Copy columns from source to target
    let copyColumns (columnNames:seq<string*string>) (mlc:MLContext) =
        mlc.Transforms.CopyColumns(columnNames |> Seq.map (fun (a,b) -> struct (a,b)) |> Seq.toArray )        
        

    /// Normalize the columns according to spezified mode
    let normalize mode outputColumnNames inputColumnNames (mlc:MLContext) =
        mlc.Transforms.Normalize(outputColumnNames,inputColumnNames,mode)
    
    module Categorical =

        /// Converts a text column into one-hot encoded vector 
        let oneHotEncoding outputColumnName inputColumnName (mlc:MLContext) =
            mlc.Transforms.Categorical.OneHotEncoding(outputColumnName,inputColumnName)

        /// Converts a text column into one-hot encoded vector 
        let oneHotEncodingBag outputColumnName inputColumnName (mlc:MLContext) =
            mlc.Transforms.Categorical.OneHotEncoding(outputColumnName,inputColumnName,Transforms.Categorical.OneHotEncodingTransformer.OutputKind.Bag)

        /// Converts a text column into one-hot encoded vector 
        let oneHotEncodingBin outputColumnName inputColumnName (mlc:MLContext) =
            mlc.Transforms.Categorical.OneHotEncoding(outputColumnName,inputColumnName,Transforms.Categorical.OneHotEncodingTransformer.OutputKind.Bin)
        
        /// Converts a text column into one-hot encoded vector 
        let oneHotEncodingInd outputColumnName inputColumnName (mlc:MLContext) =
            mlc.Transforms.Categorical.OneHotEncoding(outputColumnName,inputColumnName,Transforms.Categorical.OneHotEncodingTransformer.OutputKind.Ind)

        /// Converts a text column into one-hot encoded vector 
        let oneHotEncodingKey outputColumnName inputColumnName (mlc:MLContext) =
            mlc.Transforms.Categorical.OneHotEncoding(outputColumnName,inputColumnName,Transforms.Categorical.OneHotEncodingTransformer.OutputKind.Key)



