// https://rosettacode.org/wiki/Split_a_character_string_based_on_change_of_character#PascalABC.NET

##
var s := 'gHHH5YY++///\';
s.AdjacentGroup.Select(a -> a.JoinToString).JoinToString(', ').Print
