# docker_project
Machine Learning , Flask , Face recognition and container based docker project 
<div><h2><b>Pre-Requesites</b></h2>
<ul>
  <l1>Tested OS RHEL8 on Bridge Network</li>
  <li>Firewall Should be Started or else port forwarding wont work</li>
  <li>Docker Should be downloaded</li>
  <li>Docker Compose should be installed , code for it is given below</li>
</ul></div>
<h2>Use of this Project</h2><p>This project is made for the docker training by IIEC RISE community<br>The overall use case and meaning of my project is to create websites<br>where you dont have to remember and live in the fear forgetting passwords<br>or usernames.<br>With this service you can login and signup with your FACE</p>

<h2><b>Simple steps to create your own website which signs up and login with your face not password and username.</b></h2>

<h3>Step 1:</h3>

Download my docker image with this cmd :
```docker pull cyberwizard1/cyberwizard:6```

<h3>Step 2:</h3>

Download the only file in this repo which is : 
```docker-compose.yml```

<h3>Step 3:</h3>

Attach to the container for ip changes and starting server
```docker ps && docker attach <container name>```

<h3>Step 4:</h3>

Now got templates folder inside webapp folder
```cd /webapp/templates/```

<h3>Step 5:</h3>

Now open the 3 files 1 by 1 and just change the IP which is in 
```form action ="" ```  with your OS IP

<h3>Step 6:</h3>

Now exit the editor and type 
```cd /webapp/  && python3 index.py```

<h2>Thats it webserver is ready</h2> 

Open browser an type 
```<your ip>:5000```
