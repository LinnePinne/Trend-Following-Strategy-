# Trend-Following-Strategy

Hypotes

En snabb moving average som är högre än en långsam moving average visar stark bullish trend, ifall dessa villkor är sanna kan trade möjligheter identifieras vid pullbacks (givet att villkoren hålls sanna även fram till signal). Under dessa trendande regimer kan vi utnyttja trenden genom att gå in i nya positioner ju fler pullbacks vi har, ju högre priset stiger. På så sätt håller vi vinster så länge som möjligt och stänger endast när villkoren för trenden inte längre stämmer. Jag tror vi kommer se att strategin presterar under trendande regimer. 

Signaler

Long Signal: Fast MA över Slow MA, föregående bar stänger under Fast, nuvarande bar stänger ovanför (pullback) 

Short signal: Fast MA under Slow MA, föregående bar stängde över Fast, nuvarande bar stänger under (pullback). 

Exits

För longs: När pris går under slow MA, bullish trenden håller inte längre 

För shorts: När pris går över slow MA, bearish trenden håller inte längre 

Dessa exits fungerar som både stop loss och take profit samtidigt, utnyttjar starka trender så att vinster blir så stora som möjligt 

Möjliga filter

ADX 
