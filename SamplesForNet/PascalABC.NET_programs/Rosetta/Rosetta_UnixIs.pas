// https://rosettacode.org/wiki/Unix/ls#PascalABC.NET

##
var dirName := $'.';
(EnumerateDirectories(dirName) + EnumerateFiles(dirName)).Select(s -> s[dirName.Length+2:]).PrintLines;