// https://rosettacode.org/wiki/Reflection/List_methods#PascalABC.NET

##
uses System, System.Reflection;

var flags := BindingFlags.Instance or BindingFlags.Static
            or BindingFlags.Public or BindingFlags.DeclaredOnly;
            
typeof(integer).GetMethods(flags).PrintLines;


