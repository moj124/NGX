

var eventBox = document.getElementsByClassName('time')

// const countdownBox = document.getElementById('countdown-box')
 console.log(eventBox)

// const group_elements[i] = Date.parse(eventBox.textContent) + 5*60*1000
//  console.log(group_elements[i])

var group_elements = document.getElementsByClassName('time');
// var group_elements = document.querySelectorAll('.time');
console.log(group_elements)
var myCountdown =setInterval(()=>{
    const countdownBox = document.querySelectorAll('#countdown-box')
    var j = 0
    for (var i = 0; i < group_elements.length; i++) {
        // console.log('group elements: ', group_elements[i].textContent);
        var eventDate = Date.parse(group_elements[i].textContent) 
        console.log('date',group_elements[i].textContent)
        var now = new Date().getTime()
        // console.log(now)
        eventDate += 15 * 60 * 1000
        const dff = eventDate  - now
        console.log('diff',group_elements[i].textContent,new Date(),dff)

        // now_


        // const d = Math.floor(eventDate / (1000 * 60 * 60 * 24)-(now/(1000 * 60 * 60 * 24)))
        // const h = Math.floor((eventDate / (1000 * 60 * 60 )-(now/(1000 * 60 * 60 )))% 24)
        const m = Math.floor((eventDate / (1000 * 60)-(now/(1000 * 60  )))% 60)
        const s = Math.floor((eventDate / (1000)-(now/(1000)))% 60)
        // console.log(m)
        // console.log(s)
        // console.log(h)
        // console.log(d)
        // const countdownBox = document.getElementById('countdown-box')
        if (dff > 0){
            if (m < 10){
                if (s < 10){
                countdownBox[j].innerHTML = '0' + m +':0' + s
                }
                else{
                    countdownBox[j].innerHTML =  '0' + m +':' + s
                }
            }
            else{
                if (s < 10){
                    countdownBox[j].innerHTML =  m + ':0' + s
                }
                else{
                    countdownBox[j].innerHTML =  m + ':' + s
                }
            }
           
            
        } else{
            if (i == group_elements.length-1){
                clearInterval(myCountdown)
            }
            const items = document.getElementsByClassName('item')
            
            // countdownBox[j].innerHTML  = "countdown completed"
            items[j].style.display='none';
            location.reload()
            // for (var j = 0; j < items.length; j++) {
            //     const count = document.querySelectorAll('#countdown-box')
            //     if(count.textContent == "countdown completed"){
            //         items[j].style.display='none';
            //     }
                
            // }
            
        }
        j+=1
    }
} , 1000)








