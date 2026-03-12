uses GraphWPF,Controls,ABCDatabases;

begin
  var Страны := ЗаполнитьМассивСтран;
  var CтраныАфрики := Страны.Where(страна → страна.Континент = 'Африка')
                            .OrderBy(страна → страна.Название)
                            .Select(страна → new class(страна.Название,страна.Столица,страна.Население));
  
  LeftPanel(200,Colors.Orange);
  var ОкноСписка := SetMainControl.AsListView;
  
  Button('Все страны').Click := () -> ОкноСписка.Fill(Страны);
  Button('Cтраны Африки по алфавиту').Click := () -> ОкноСписка.Fill(CтраныАфрики);
end.

