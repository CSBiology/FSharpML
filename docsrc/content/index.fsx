(*** hide ***)
// This block of code is omitted in the generated HTML documentation. Use 
// it to define helpers that you do not want to show in the documentation.
#r "../../packages/formatting/Newtonsoft.Json/lib/netstandard2.0/Newtonsoft.Json.dll"
#r @"../../packages/formatting/FSharp.Plotly/lib/netstandard2.0/FSharp.Plotly.dll" 
#I "../../"
open FSharp.Plotly
//open FSharp.Plotly

(**
FSharpML: Explore ML.Net in F#
==============================

FSharpML is a lightweight API writen in F# on top of the powerful machine learning framework [ML.Net](http://dot.net/ml) library. It is designed to enable users to explore ML.Net in a scriptable manner and maintaining the functional style of F#.
The samples are ported from the official site [Samples for ML.NET](https://github.com/dotnet/machinelearning-samples).

After installing the package via Nuget we can load the delivered reference script and start using ML.Net in conjunction with FSharpML.
*)



// STEP 1: Common data loading configuration
let chart1 = Chart.Point([1 .. 10], [1 .. 10])
(***include-value:chart1***)



