(* ::Package:: *)

BeginPackage[ "plottingfunctions`"]



CobwebPlot::usage = 
    "CobWebPlot[f_,start_,nIter_,xrange_] makes a cobwebplot"
  
  Begin[ "Private`"]
  ClearAll[CobwebPlot]
  Options[CobwebPlot]=Join[{CobStyle->Automatic},Options[Graphics]];
 CobwebPlot[f_,start_?NumericQ,n_,xrange:{xmin_,xmax_},opts:OptionsPattern[]]:=Module[{cob,x,g1,coor},
  cob=NestList[f,N[start],n];
  coor = Partition[Riffle[cob,cob],2,1];
  coor[[1,2]]=0;
  cobstyle=OptionValue[CobwebPlot,CobStyle];
  cobstyle=If[cobstyle===Automatic,Red,cobstyle];
  g1=Graphics[{cobstyle,Line[coor]}];
Show[{Plot[{x,f[x]},{x,xmin,xmax},PlotStyle->{{Thick,Black},Black}],g1},FilterRules[{opts},Options[Graphics]]]
]

  End[]

  EndPackage[]
