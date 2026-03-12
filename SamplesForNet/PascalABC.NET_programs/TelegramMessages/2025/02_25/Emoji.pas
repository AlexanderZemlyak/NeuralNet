// Задача 5: Переводчик эмодзи
// Описание: Программа переводит текстовые сообщения в эмодзи
begin
  var emoji := Dict(
    'улыбка' to $'😊',
    'солнце' to $'☀',
    'дождь' to $'🌧',
    'кот' to $'🐱',
    'сердце' to $'❤'
  );
  
  var message := 'Я и кот, солнце в душе и прекрасная улыбка в моем сердце';
  
  Println('Перевод сообщения:');
  foreach var word in message.ToWords do
    if word in emoji then
      Print(emoji[word])
    else
      Print(word);
end.