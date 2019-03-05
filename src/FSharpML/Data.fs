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
        open System.Reflection
        open Microsoft.FSharp.Reflection        

        let createColumn name datakind index =
            TextLoader.Column(name,Nullable datakind,index)
    

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
                    | Some attrib -> attrib.Name
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
                    | t when t = typeof<SByte>              -> DataKind.I1  |> Some
                    | t when t = typeof<Byte>               -> DataKind.U1  |> Some
                    | t when t = typeof<Int16>              -> DataKind.I2  |> Some
                    | t when t = typeof<UInt16>             -> DataKind.U2  |> Some
                    | t when t = typeof<Int32>              -> DataKind.I4  |> Some
                    | t when t = typeof<UInt32>             -> DataKind.U4  |> Some
                    | t when t = typeof<Int64>              -> DataKind.I8  |> Some
                    | t when t = typeof<UInt64>             -> DataKind.U8  |> Some
                    | t when t = typeof<Single>             -> DataKind.R4  |> Some
                    | t when t = typeof<Double>             -> DataKind.R8  |> Some
                    | t when t = typeof<String>             -> DataKind.TX  |> Some
                    | t when t = typeof<Boolean>            -> DataKind.BL  |> Some
                    | t when t = typeof<TimeSpan>           -> DataKind.TS  |> Some
                    | t when t = typeof<DateTime>           -> DataKind.DT  |> Some
                    | t when t = typeof<DateTimeOffset>     -> DataKind.DZ  |> Some
            
                    | t -> None //failwithf "a System.Type %A has no corresponding DataKind value" t 

        
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
                    tlc.Type   <- tryGetDataKind field.PropertyType |> Option.toNullable
                    Some (tlc)
                ) 
            |> Array.choose id

            
    
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
    
    /// Reads a data view from text file as stream input
    let readFromTextStream (mlc:MLContext) separatorChar hasHeader columns (stream:IMultiStreamSource) = 
        let txtld =
            new TextLoader(
                mlc,
                columns = columns,
                hasHeader = hasHeader,
                separatorChar = separatorChar)
        
        txtld.Read(stream)
            

    /// Reads a data view from binary file
    let readFromBinary (mlc:MLContext) (multiStream: IMultiStreamSource) = 
        mlc.Data. ReadFromBinary(multiStream)
    
    /// Saves data view to text file
    let saveAsText (mlc:MLContext) separatorChar hasHeader  (stream:System.IO.Stream) (data:IDataView) = 
        mlc.Data.SaveAsText(data, stream, separatorChar, hasHeader)

    /// Saves data view to binary file
    let saveAsBinary (mlc:MLContext) (stream:System.IO.Stream) (data:IDataView) = 
        mlc.Data.SaveAsBinary(data, stream)

    
    //let add () = ()
    //    let advb = Microsoft.ML.Data.ArrayDataViewBuilder(mlContext)
    //    advb.AddColumn<bool>("Label",bt,[|ls|])
    //    advb.AddColumn<ReadOnlyMemory<char>>("Text",tt,[|ts|])
    //    let dataView = advb.GetDataView()
    

