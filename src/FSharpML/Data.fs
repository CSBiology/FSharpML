namespace FSharpML

open System
open Microsoft.ML
open Microsoft.ML.Core.Data
open Microsoft.Data.DataView
open Microsoft.ML.Data
open System.Runtime.InteropServices


// Maybe name Data according to Microsoft.ML.Data
module Data =
    
    module DataView =
    
    /// Returns column names
        let getColumnNames (data:IDataView) =
            data.Schema |> Seq.map (fun c -> c.Name) |> Seq.toArray
    
        /// Try get the number of rows
        let tryGetRowCount (data:IDataView) =
            data.GetRowCount() |> Option.ofNullable
        
        /// Peek max of row 
        let preview maxRow (data:IDataView) =
            data.Preview(maxRow)
            
    module TextLoader =
        
        let createColumn name datakind index =
            TextLoader.Column(name,Nullable datakind,index)
    
    
    /// Returns all values of a column
    let getColumn<'a> (mlc:MLContext) columnName (data:IDataView) = 
        data.GetColumn<'a>(mlc,columnName)
    
    /// Keeps only those rows that are between lower and upper range condition
    let filterByColumn (mlc:MLContext) columnName lower upper (data:IDataView) = 
        mlc.Data.FilterByColumn(data,columnName,lower,upper)
        
    /// Creates seq<'Trow> from data view
    let createEnumerable<'Trow when 'Trow:(new : unit -> 'Trow) and 'Trow :not struct> (mlc:MLContext) (data:IDataView) = 
        mlc.CreateEnumerable<'Trow>(data,false)

    /// Reads a data view from seq<'Trow> 
    let readFromEnumerable<'Trow when 'Trow:(new : unit -> 'Trow) and 'Trow :not struct> (mlc:MLContext) (data) = 
        mlc.Data.ReadFromEnumerable<'Trow>(data)
    
    /// Reads a data view from text file
    let readFromTextFile (mlc:MLContext) separatorChar hasHeader columns path = 
        mlc.Data.ReadFromTextFile( 
            path = path,
            columns = columns,
            hasHeader = hasHeader,
            separatorChar = separatorChar)
    
    /// Reads a data view from binary file
    let readFromBinary (mlc:MLContext) (multiStream: IMultiStreamSource) = 
        mlc.Data. ReadFromBinary(multiStream)
    
    /// Saves data view to text file
    let saveAsText (mlc:MLContext) separatorChar hasHeader  (stream:System.IO.Stream) (data:IDataView) = 
        mlc.Data.SaveAsText(data, stream, separatorChar, hasHeader)

    /// Saves data view to binary file
    let saveAsBinary (mlc:MLContext) (stream:System.IO.Stream) (data:IDataView) = 
        mlc.Data.SaveAsBinary(data, stream)



    