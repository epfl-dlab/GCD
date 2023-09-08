# #1545
# demo1 = {
#     "text": "The excess supply pushed gasoline prices down in that period" ,
#     "output": "[ ( S ( NP-SBJ ( DT The ) ( JJ excess ) ( NN supply ) ) ( VP ( VBD pushed ) ( NP ( NN gasoline ) ( NNS prices ) ) ( ADVP-DIR ( RP down ) ) ( PP-TMP ( IN in ) ( NP ( DT that ) ( NN period ) ) ) ) ) ]",
# }
#
# #543
# demo2 = {
#     "text": "The spacecraft 's five astronauts are to dispatch the Galileo space probe on an exploration mission to Jupiter",
#     "output": "[ ( S ( NP-SBJ-1 ( NP ( DT The ) ( NN spacecraft ) ( POS 's ) ) ( CD five ) ( NNS astronauts ) ) ( VP ( VBP are ) ( S ( VP ( TO to ) ( VP ( VB dispatch ) ( NP ( DT the ) ( NNP Galileo ) ( NN space ) ( NN probe ) ) ( PP-CLR ( IN on ) ( NP ( NP ( DT an ) ( NN exploration ) ( NN mission ) ) ( PP ( TO to ) ( NP ( NNP Jupiter ) ) ) ) ) ) ) ) ) ) ]",
# }
#
# # 803
# demo3 = {
#     "text": "Or is triskaidekaphobia fear of the number 13 justified",
#     "output": "[ ( SQ ( CC Or ) ( VBZ is ) ( NP-SBJ ( NP ( NN triskaidekaphobia ) ) ( PRN ( NP ( NP ( NN fear ) ) ( PP ( IN of ) ( NP ( NP ( DT the ) ( NN number ) ) ( NP ( CD 13 ) ) ) ) ) ) ) ( ADJP-PRD ( VBN justified ) ) ) ]",
# }
#
# # 312
# demo4 = {
#     "text": "Restrict the ability of real estate owners to escape taxes by swapping one piece of property for another instead of selling it for cash",
#     "output": "[ ( FRAG ( VP ( VB Restrict ) ( NP ( NP ( DT the ) ( NN ability ) ) ( PP ( IN of ) ( NP ( JJ real ) ( NN estate ) ( NNS owners ) ) ) ( S-1 ( VP ( TO to ) ( VP ( VB escape ) ( NP ( NNS taxes ) ) ( PP-MNR ( IN by ) ( S-NOM ( VP ( VBG swapping ) ( NP ( NP ( CD one ) ( NN piece ) ) ( PP ( IN of ) ( NP ( NN property ) ) ) ) ( PP-CLR ( IN for ) ( NP ( DT another ) ) ) ( PP ( RB instead ) ( IN of ) ( S-NOM ( VP ( VBG selling ) ( NP ( PRP it ) ) ( PP-CLR ( IN for ) ( NP ( NN cash ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ]",
# }
#
# # 1354
# demo5 = {
#     "text": "Pension funds insurers and other behemoths of the investing world said they began scooping up stocks during Friday 's market rout",
#     "output": "[ ( S ( NP-SBJ ( NP ( NN Pension ) ( NNS funds ) ) ( NP ( NNS insurers ) ) ( CC and ) ( NP ( NP ( JJ other ) ( NNS behemoths ) ) ( PP ( IN of ) ( NP ( DT the ) ( NN investing ) ( NN world ) ) ) ) ) ( VP ( VBD said ) ( SBAR ( S ( NP-SBJ-1 ( PRP they ) ) ( VP ( VBD began ) ( S ( VP ( VBG scooping ) ( PRT ( RP up ) ) ( NP ( NNS stocks ) ) ( PP-TMP ( IN during ) ( NP ( NP ( NNP Friday ) ( POS 's ) ) ( NN market ) ( NN rout ) ) ) ) ) ) ) ) ) ) ]"
# }
#
# # 1797
# demo6 = {
#     "text": "A month ago Hertz of Park Ridge N.J. said that it would drop marketing agreements at year end with Delta America West and Texas Air Corp. 's Continental Airlines and Eastern Airlines and that pacts with American Airlines UAL Inc 's United Airlines and USAir also would be ended sometime after Dec. 31",
#     "output": "[ ( S ( ADVP-TMP ( NP ( DT A ) ( NN month ) ) ( RB ago ) ) ( NP-SBJ ( NP ( NNP Hertz ) ) ( PP ( IN of ) ( NP ( NP ( NNP Park ) ( NNP Ridge ) ) ( NP ( NNP N.J. ) ) ) ) ) ( VP ( VBD said ) ( SBAR ( SBAR ( IN that ) ( S ( NP-SBJ ( PRP it ) ) ( VP ( MD would ) ( VP ( VB drop ) ( NP ( NP ( PRP$ its ) ( NN marketing ) ( NNS agreements ) ) ) ( PP-TMP ( IN at ) ( NP ( NN year ) ( NN end ) ) ) ( PP-1 ( IN with ) ( NP ( NP ( NNP Delta ) ) ( NP ( NNP America ) ( NNP West ) ) ( CC and ) ( NP ( NP ( NNP Texas ) ( NNP Air ) ( NNP Corp. ) ( POS 's ) ) ( NX ( NX ( NNP Continental ) ( NNP Airlines ) ) ( CC and ) ( NX ( NNP Eastern ) ( NNP Airlines ) ) ) ) ) ) ) ) ) ) ( CC and ) ( SBAR ( IN that ) ( S ( NP-SBJ-2 ( NP ( NNS pacts ) ) ( PP ( IN with ) ( NP ( NP ( NNP American ) ( NNP Airlines ) ) ( NP ( NP ( NNP UAL ) ( NNP Inc ) ( POS 's ) ) ( NNP United ) ( NNP Airlines ) ) ( CC and ) ( NP ( NNP USAir ) ) ) ) ) ( ADVP ( RB also ) ) ( VP ( MD would ) ( VP ( VB be ) ( VP ( VBN ended ) ( ADVP-TMP ( RB sometime ) ( PP ( IN after ) ( NP ( NNP Dec. ) ( CD 31 ) ) ) ) ) ) ) ) ) ) ) ) ]",
# }
#
# # 1347
# demo7 = {
#     "text": "There has n't been any fundamental change in the economy added John Smale Procter & Gamble Co. took an 8.75 slide to close at 120.75",
#     "output": "[ ( SINV ( S-TPC-1 ( NP-SBJ ( RB There ) ) ( VP ( VBZ has ) ( RB n't ) ( VP ( VBN been ) ( NP-PRD ( NP ( DT any ) ( JJ fundamental ) ( NN change ) ) ( PP-LOC ( IN in ) ( NP ( DT the ) ( NN economy ) ) ) ) ) ) ) ( VP ( VBD added ) ) ( NP-SBJ ( NP ( NNP John ) ( NNP Smale ) ) ( SBAR ( WHNP ( WP$ whose ) ) ( S ( NP-SBJ-2 ( NNP Procter ) ( CC & ) ( NNP Gamble ) ( NNP Co. ) ) ( VP ( VBD took ) ( NP ( DT an ) ( ADJP ( CD 8.75 ) ) ( NN slide ) ) ( S-CLR ( VP ( TO to ) ( VP ( VB close ) ( PP-CLR ( IN at ) ( NP ( CD 120.75 ) ) ) ) ) ) ) ) ) ) ) ]",
# }
#
# # 2267
#
# demo8 = {
#     "text": "Nevertheless Tandem faces a variety of challenges the biggest being that customers generally view the company 's computers as complementary to IBM 's mainframes",
#     "output": "[ ( S ( ADVP ( RB Nevertheless ) ) ( NP-SBJ ( NNP Tandem ) ) ( VP ( VBZ faces ) ( NP ( NP ( DT a ) ( NN variety ) ) ( PP ( IN of ) ( NP ( NP ( NNS challenges ) ) ( S ( NP-SBJ ( DT the ) ( JJS biggest ) ) ( VP ( NN being ) ( SBAR-PRD ( IN that ) ( S ( NP-SBJ ( NNS customers ) ) ( ADVP ( RB generally ) ) ( VP ( VBP view ) ( NP ( NP ( DT the ) ( NN company ) ( POS 's ) ) ( NNS computers ) ) ( PP-CLR ( IN as ) ( ADJP ( JJ complementary ) ( PP ( TO to ) ( NP ( NP ( NNP IBM ) ( POS 's ) ) ( NNS mainframes ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ]"
# }

demo1 = {}
demo2 = {}
demo3 = {}
demo4 = {}
demo5 = {}
demo6 = {}
demo7 = {}
demo8 = {}


DEMOs = [demo1, demo2, demo3, demo4, demo5, demo6, demo7, demo8]
