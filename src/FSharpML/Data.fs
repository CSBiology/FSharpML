namespace FSharpML

open System
open Microsoft.ML
open Microsoft.ML.Data
//open Microsoft.ML.Core.Data
//open Microsoft.Data.DataView
open Microsoft.ML.Data
open System.Runtime.InteropServices




module DefaultColumnNames =

    let Label = "Label"
    let Score = "Score"
    let Probability = "Probability"
    let PredictedLabel = "PredictedLabel"
    let Features = "Features"


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
        open System.Reflection
        open Microsoft.FSharp.Reflection        

        let createColumn name datakind index =
            TextLoader.Column(name,datakind,index)
    

        /// Returns given attribute from property info as optional 
        let private tryGetCustomAttribute<'a> (findAncestor:bool) (propInfo :PropertyInfo) =   
            let attributeType = typeof<'a>
            let attrib = propInfo.GetCustomAttribute(attributeType, findAncestor)
            match box attrib with
            | (:? 'a) as customAttribute -> Some(unbox<'a> customAttribute)
            | _ -> None

        /// Returns Schema as an array of SchemaItems 
        let columnsFrom (schemaType)=

            let tryGetFieldValue (instance:obj) (fieldName:string) =

                let bindFlags = BindingFlags.Instance ||| BindingFlags.Public ||| BindingFlags.NonPublic ||| BindingFlags.Static
                let field = instance.GetType().GetField(fieldName, bindFlags)
                match obj.ReferenceEquals(field, null) with
                | false -> field.GetValue(instance) |> Some
                | true  -> None        
        
            let isIgnored (propertyInfo:PropertyInfo) = 
                match (tryGetCustomAttribute<NoColumnAttribute> false propertyInfo) with
                    | Some attrib -> true
                    | _           -> false

            let getColumnName (propertyInfo:PropertyInfo) =
                match (tryGetCustomAttribute<ColumnNameAttribute> false propertyInfo) with
                    | Some attrib -> string attrib.TypeId //Name
                    | _           -> propertyInfo.Name

            let getColumnIndexOrDefault (defaultIndex:int) (propertyInfo:PropertyInfo) =
                match (tryGetCustomAttribute<LoadColumnAttribute> false propertyInfo) with
                    | Some attrib -> match tryGetFieldValue attrib "Sources" with
                                     | Some sources -> sources :?> seq<TextLoader.Range> |> Seq.toArray
                                     | None         ->  failwithf "LoadColumnAttribute does not contain a Source field"
                    | _           -> [|TextLoader.Range(defaultIndex)|]

            /// Try to map a System.Type to a corresponding DataKind value.        
            let tryGetDataKind _type =
                    match _type with
                    | t when t = typeof<SByte>              -> DataKind.SByte          //|> Some
                    | t when t = typeof<Byte>               -> DataKind.Byte           //|> Some
                    | t when t = typeof<Int16>              -> DataKind.Int16          //|> Some
                    | t when t = typeof<UInt16>             -> DataKind.UInt16         //|> Some
                    | t when t = typeof<Int32>              -> DataKind.Int32          //|> Some
                    | t when t = typeof<UInt32>             -> DataKind.UInt32         //|> Some
                    | t when t = typeof<Int64>              -> DataKind.Int64          //|> Some
                    | t when t = typeof<UInt64>             -> DataKind.UInt64         //|> Some
                    | t when t = typeof<Single>             -> DataKind.Single         //|> Some
                    | t when t = typeof<Double>             -> DataKind.Double         //|> Some
                    | t when t = typeof<String>             -> DataKind.String         //|> Some
                    | t when t = typeof<Boolean>            -> DataKind.Boolean        //|> Some
                    | t when t = typeof<TimeSpan>           -> DataKind.TimeSpan       //|> Some
                    | t when t = typeof<DateTime>           -> DataKind.DateTime       //|> Some
                    | t when t = typeof<DateTimeOffset>     -> DataKind.DateTimeOffset //|> Some
            
                    | t -> failwithf "a System.Type %A has no corresponding DataKind value" t 

        
            // Grab the object for the type that describes the schema
            //let schemaType = typeof<'Schema>
            // Grab the fields from that type
            let fields = FSharpType.GetRecordFields(schemaType)

            fields 
            |> Array.mapi( fun fieldIndex field -> 
                if isIgnored field then
                    None
                else
                    let colName  = getColumnName field
                    let colIndex = getColumnIndexOrDefault fieldIndex field
                    let tlc = TextLoader.Column()
                    tlc.Name   <- colName
                    tlc.Source <- colIndex
                    tlc.DataKind <- tryGetDataKind field.PropertyType
                    //tlc.Type   <- tryGetDataKind field.PropertyType |> Option.toNullable
                    Some (tlc)
                ) 
            |> Array.choose id

            
    
    /// Returns all values of a column
    let getColumn<'a> (columnName:string) (data:IDataView) =         
        data.GetColumn<'a>(columnName)
    
    // Keeps only those rows that are between lower and upper range condition
    let filterByColumn (mlc:MLContext) columnName lower upper (data:IDataView) =         
        mlc.Data.FilterRowsByColumn(data,columnName,lower,upper)
        
    /// Creates seq<'Trow> from data view
    let createEnumerable<'Trow when 'Trow:(new : unit -> 'Trow) and 'Trow :not struct> (mlc:MLContext) (data:IDataView) =         
        mlc.Data.CreateEnumerable<'Trow>(data,false)

    /// Reads a data view from seq<'Trow> 
    let loadFromEnumerable<'Trow when 'Trow:(new : unit -> 'Trow) and 'Trow :not struct> (mlc:MLContext) (data) = 
        mlc.Data.LoadFromEnumerable<'Trow>(data)
    
    /// Reads a data view from text file
    let loadFromTextFile (mlc:MLContext) separatorChar hasHeader columns path = 
        mlc.Data.LoadFromTextFile( 
            path = path,
            columns = columns,
            hasHeader = hasHeader,
            separatorChar = separatorChar)
    
    ///// Reads a data view from text file as stream input
    //let loadFromTextStream (mlc:MLContext) separatorChar hasHeader columns (stream:IMultiStreamSource) = 
        
    //    let txtld = mlc.Data.CreateTextLoader()
    //    //let txtld =
    //    //    new TextLoader(
    //    //        mlc,
    //    //        columns = columns,
    //    //        hasHeader = hasHeader,
    //    //        separatorChar = separatorChar)
        
    //    txtld.Load(stream)
            

    /// Reads a data view from binary file
    let loadFromBinary (mlc:MLContext) (multiStream: IMultiStreamSource) = 
        mlc.Data.LoadFromBinary(multiStream)
    
    /// Saves data view to text file
    let saveAsText (mlc:MLContext) separatorChar hasHeader  (stream:System.IO.Stream) (data:IDataView) = 
        mlc.Data.SaveAsText(data, stream, separatorChar, hasHeader)

    /// Saves data view to binary file
    let saveAsBinary (mlc:MLContext) (stream:System.IO.Stream) (data:IDataView) = 
        mlc.Data.SaveAsBinary(data, stream)

    
    ////let add () = ()
    ////    let advb = Microsoft.ML.Data.ArrayDataViewBuilder(mlContext)
    ////    advb.AddColumn<bool>("Label",bt,[|ls|])
    ////    advb.AddColumn<ReadOnlyMemory<char>>("Text",tt,[|ts|])
    ////    let dataView = advb.GetDataView()
    

