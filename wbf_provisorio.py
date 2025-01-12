"""wbf con gpu provisorio"""

import time
import numpy as np
import pyopencl as cl
import pygame
import crear_diccionario

V = pygame.Vector2

game = crear_diccionario

num_balls = len(game.balls_dict)
grid_size = int(np.sqrt(num_balls/3))
cell_max_capacity = int(20*num_balls/grid_size**2)

print("grid size",grid_size)

class Work_in_GPU():
    """"worh bwtweeen frams"""
    def __init__(self) -> ():
        
        # self.time = time.perf_counter()
        # self.dt = time.perf_counter()

        self.time = 0.0001
        self.dt = 0.0001

        #OpenCL enviroment config
        self.platform = cl.get_platforms()[0]
        self.device = self.platform.get_devices()[0]
        self.context = cl.Context([self.device])
        self.queue = cl.CommandQueue(self.context)

        self.lets_work()

    def lets_work(self):

        compilate1 = time.perf_counter()
        self.gpu_kernel = self.kernel_code_compile()
        compilate2 = time.perf_counter()
        input("despues de compilar kernel")
        laad1 = time.perf_counter()
        self.load_buffer(game)
        laad2 = time.perf_counter()
        print("tiempo compilacion:",compilate2-compilate1)
        print("tiempo carga:",laad2-laad1)
        # input("despues de load buffer")

        for i in range(1):
            
            inicio = time.perf_counter()   
            
            
            fill1 = time.perf_counter()
            self.fill_grid(game)
            fill2 = time.perf_counter()
            # input("despues de fill grid")

            sweep1 = time.perf_counter()
            self.sweep_grid(game)
            sweep2 = time.perf_counter()
            # input("despues de sweep grid")

            move1 = time.perf_counter()
            self.move_balls(game)
            move2 = time.perf_counter()
            # input("despues de move balls")

            fin = time.perf_counter()

            print("tiempo move",move2-move1,"tiempo fill",fill2-fill1,"tiempo sweep",sweep2-sweep1)
            print("     %move:",100*(move2-move1)/(fin-inicio),"%fill:",100*(fill2-fill1)/(fin-inicio),"%sweep:",100*(sweep2-sweep1)/(fin-inicio))
            print("total:",move2-move1+fill2-fill1+sweep2-sweep1)
        
        input("despues de 3 funciones")
        self.download_buffers()
        input("despues de download buffers")
        self.grid_view(game)



    def grid_view(self,game):

        num_balls = len(game.balls_dict.keys())
        # grid_size = 400
        # cell_max_capacity = int(200*num_balls/grid_size**2) #200 times the mean
        grid_reshaped = self.grid_array.reshape((grid_size, grid_size, cell_max_capacity))
        print("----------grid--despues reshape----------------------------")
        print(type(grid_reshaped),grid_reshaped.shape)
        # Mostrar los primeros valores de cada celda para ver los IDs de las bolas
        for i in range(grid_size):
            for j in range(grid_size):
                # Mostrar los primeros 10 IDs de cada celda
                print(f"Celda ({i}, {j}): {grid_reshaped[i, j, :20]}")


    def load_buffer(self,game):
        """load the buffer with the current state of the game"""

        ################load balls buffer#########################
        self.balls_array = np.array([(key,
                                    ball.p[0],    ball.p[1],
                                    ball.v[0],    ball.v[1],
                                    ball.att[0] , ball.att[1],
                                    ball.radius)
                                    for key, ball in game.balls_dict.items()] ,
                                    dtype=np.float32).astype(np.float32)
        # print("-------------------balls array------------")
        # print(type(self.balls_array),self.balls_array.shape)
        # print(self.balls_array)
        # input("pausa")

        
        self.balls_buffer = cl.Buffer(self.context, cl.mem_flags.READ_ONLY |
                                        cl.mem_flags.COPY_HOST_PTR, hostbuf=self.balls_array)
        
        ############load balls corners buffer########################



        self.balls_corners_array = np.array([
            [key,
                ball.rect.topleft[0], ball.rect.bottomright[0],
                ball.rect.topleft[1], ball.rect.bottomright[1]] 
            for key, ball in game.balls_dict.items()
            ], dtype=np.int32).astype(np.int32)
        
        # print("-------------------balls corners array------------")
        # print(type(self.balls_corners_array),self.balls_corners_array.shape)
        # print(self.balls_corners_array)
        # input("pausa")

        
        self.balls_corners_buffer = cl.Buffer(self.context, cl.mem_flags.READ_ONLY |
                                        cl.mem_flags.COPY_HOST_PTR, hostbuf=self.balls_corners_array)                                            

        #################load grid buffer###########################
        num_balls = len(game.balls_dict.keys())
        # grid_size = 400
        # cell_max_capacity = int(200*num_balls/grid_size**2) #200 times the mean
        area = [800,800]
        
        cell_size = area[0]/grid_size

        self.grid_array = np.full((grid_size * grid_size * cell_max_capacity), -1, dtype=np.int32).astype(np.int32)

                          
        self.grid_buffer = cl.Buffer(self.context, cl.mem_flags.READ_WRITE |
                                        cl.mem_flags.COPY_HOST_PTR, hostbuf=self.grid_array)
        
        #################load overlaps buffer###########################

        max_expected_overlaps = len(game.balls_dict) #de sobra

        self.ovelaps_array = np.full((max_expected_overlaps*2), -1, dtype=np.int32).astype(np.int32)

        self.overlaps_buffer = cl.Buffer(self.context, cl.mem_flags.READ_WRITE |
                                        cl.mem_flags.COPY_HOST_PTR, hostbuf=self.ovelaps_array)
        
    ##kernel code compile.move balls, fill_grid,sweep grid######
    def kernel_code_compile(self):

        gpu_kernel_code= """

        //################ move balls kernel ############################
        

        __kernel void move_balls(__global float *balls, float time){

            int ball_index = get_global_id(0);
            float x = (float) balls[ball_index * 8 + 1];
            float y = (float) balls[ball_index * 8 + 2];
            float vx = (float) balls[ball_index * 8 + 3];
            float vy = (float) balls[ball_index * 8 + 4];
            float att_x = (float) balls[ball_index * 8 + 5];
            float att_y = (float) balls[ball_index * 8 + 6];
            //if (ball_index < 1){
            //    printf("thread :%d,pos x:%f,vx:%f,tiempo:%f", ball_index,x,vx,time);
            //    }
            x += vx * time + 0.5*att_x*time*time;
            y += vy * time + 0.5*att_y*time*time;
            //if (ball_index < 1){
            //    printf("thread :%d,pos x:%f", ball_index,x);
            //    }
            balls[ball_index * 8 + 1] = x;
            balls[ball_index * 8 + 2] = y;     
            balls[ball_index * 8 + 3]  = att_x*time;
            balls[ball_index * 8 + 4]  = att_y*time;        
            }

        //###################fill grid kernel############################

        __kernel void fill_grid(__global int *balls_corners, __global int *grid, int num_balls,
                                 int grid_size, int width, int height, int cell_max_capacity) {

            
            int ball_index = get_global_id(0);
            int cell_size   = width / grid_size;

            int ball_id = (int) balls_corners[ball_index * 5 + 0];
            int top     = (int) balls_corners[ball_index * 5 + 1];
            int bottom  = (int) balls_corners[ball_index * 5 + 2];
            int left    = (int) balls_corners[ball_index * 5 + 3];
            int right   = (int) balls_corners[ball_index * 5 + 4];
            //if(ball_index == 1){
            //    printf("ball_id:%d,top:%d,bottom:%d,left:%d,right:%d",ball_id,top,bottom,left,right);
            //    }

            int cell_top    = top / cell_size;
            if (cell_top < 0) {
                cell_top = 0;
            }
            int cell_bottom = bottom / cell_size;
            if (cell_bottom > grid_size - 1) {
                cell_bottom = grid_size - 1;
                }
            int cell_left   = left / cell_size;
            if (cell_left < 0) {
            cell_left = 0;
            }
            int cell_right  = right / cell_size;
            if (cell_right > grid_size - 1) {
                cell_right = grid_size - 1;
                }

            for (int i = cell_top; i <= cell_bottom; i++) {
                for (int j = cell_left; j <= cell_right; j++) {
            
                    int grid_index = (i * grid_size + j) * cell_max_capacity;

                    for (int k = 0; k < cell_max_capacity; k++) {
                        if (atomic_cmpxchg(&grid[grid_index + k], -1, ball_index) == -1) {
                            break;
                        }
                    }
                }
            }
        }

        //######### sweep grid kernel ###############################
        __kernel void sweep_grid(__global int *overlaps, __global int *grid,__global float *balls,__global int *balls_corners,
          int grid_size, int cell_max_capacity, int max_expected_overlaps,int num_balls) {

            int cell_index = get_global_id(0);

            int top1;
            int bottom1;
            int left1;
            int right1;
            int top2;
            int bottom2;
            int left2;
            int right2;
            int ball_1_index;
            int ball_2_index;
            float x1;
            float y1;
            float x2;
            float y2;
            float radius_sum;

            for (int i = 0; i < cell_max_capacity -1; i++) {

                               

                if (grid[cell_index*cell_max_capacity + i] == -1) {
                    break;
                    }
                ball_1_index = grid[cell_index*cell_max_capacity + i + 0];


            
                x1       = (float) balls[ball_1_index * 8 + 1];
                y1       = (float) balls[ball_1_index * 8 + 2];
                top1     = (int) balls_corners[ball_1_index + 1];
                bottom1  = (int) balls_corners[ball_1_index + 2];
                left1    = (int) balls_corners[ball_1_index + 3];
                right1   = (int) balls_corners[ball_1_index + 4];

                //printf("ball nr:%d,x:%f,y:%f,right:%d",ball_1_index,x1,y1,right1);       
                
                for (int j = i + 1; j < cell_max_capacity; j++){


                    if (grid[cell_index*cell_max_capacity + j] == -1) {
                        break;
                        }

                    ball_2_index = grid[cell_index*cell_max_capacity + j + 0];
                    if(ball_1_index == 1){
                        //printf("bolas chocando:bola1:%d,bolas:%d",ball_1_index,ball_2_index );
                    }
                    x2       = (float) balls[ball_2_index * 8 + 1];
                    y2       = (float) balls[ball_2_index * 8 + 2];
                    top2     = (int) balls_corners[ball_2_index + 1];
                    bottom2  = (int) balls_corners[ball_2_index + 2];
                    left2    = (int) balls_corners[ball_2_index + 3];
                    right2   = (int) balls_corners[ball_2_index + 4];

                    //printf("ball nr:%d,x:%f,y:%f,right:%d",ball_2_index,x2,y2,right2); 

                    if ((right1 > right2 && left1 < right2) || (left1 < left2 && right1 > left2)){
                    
                        if ((bottom1 > bottom2 && top1 < bottom2) || (top1 < top2 && bottom1 > top2)){

                            
                    
                            radius_sum = balls[ball_1_index * 8 + 7] + balls[ball_2_index * 8 + 7];
                            //printf("resta:%f",radius_sum-(y2 - y1)*(y2 - y1)-(x2 - x1)*(x2 - x1));
                            if (radius_sum*radius_sum > (x2 - x1)*(x2 - x1) + (y2 - y1)*(y2 - y1)) {
                                //printf("bolas chocando:bola1:%d,bolas:%d",ball_1_index,ball_2_index );

                                
                            
                                for (int k = 0; k < max_expected_overlaps*2; k = k + 2) {
                                    if (atomic_cmpxchg(&overlaps[k], -1, ball_1_index) == -1) {
                                        overlaps[k + 1] = ball_2_index;
                                        //printf("lectura overlaps:b1:%d,b2:%d",overlaps[k],overlaps[k+1]);
                                        break;
                                    }
                                }
                            }      
                        }                
                    
                    }
                }
            }
        
        }
        """

        gpu_kernel = cl.Program(self.context, gpu_kernel_code).build()
        return gpu_kernel
    
    def move_balls(self, game):
        """moves balls in the game using GPU acceleration"""
        num_balls = len(game.balls_dict)
        t = self.dt

        #Ejecutar  el kernel update
        work_group_size = 64
        self.gpu_kernel.move_balls(self.queue, (num_balls,), (work_group_size,), self.balls_buffer, np.float32(t))

    def fill_grid(self,game):
        """fill grid with balls in the game using GPU acceleration"""
        num_balls = len(game.balls_dict.keys())
        # grid_size = 400
        cell_num = grid_size**2
        # cell_max_capacity = int(200*num_balls/cell_num) #200 times the mean
        area = [800,800]
        
        cell_size = area[0]/grid_size

    # Ejecutar el kernel fill grid

        self.gpu_kernel.fill_grid(self.queue, (num_balls,), None, self.balls_corners_buffer,
                             self.grid_buffer, np.int32(num_balls),
                             np.int32(grid_size),np.int32(area[0]),np.int32(area[1]),
                             np.int32(cell_max_capacity))

    def sweep_grid(self,game):
        """sweep grid to find overlaps using GPU acceleration"""

        num_balls = len(game.balls_dict.keys())
        # grid_size = 400
        cell_num = grid_size**2
        # cell_max_capacity = int(200*num_balls/cell_num)#200 times the mean
        max_expected_overlaps = len(game.balls_dict) #de sobra

        
        
        self.gpu_kernel.sweep_grid(self.queue,(cell_num,),None,self.overlaps_buffer,self.grid_buffer,
                                   self.balls_buffer,self.balls_corners_buffer,np.int32(grid_size),
                                   np.int32(cell_max_capacity),np.int32(max_expected_overlaps),
                                   np.int32(num_balls))




    def download_buffers(self):

        cl.enqueue_copy(self.queue, self.grid_array, self.grid_buffer).wait()
        cl.enqueue_copy(self.queue,self.balls_array,self.balls_buffer).wait()
        cl.enqueue_copy(self.queue,self.ovelaps_array,self.overlaps_buffer).wait()   

        print("-------------------gid array lego download------------")
        print(type(self.grid_array),self.grid_array.shape)
        print(self.grid_array)
        input("pausa")  

        print("-------------------balls array luego download------------")
        print(type(self.balls_array),self.balls_array.shape)
        print(self.balls_array)
        input("pausa")  

    # def translate_grid(self,game):
    #     num_balls = len(game.balls_dict.keys())
    #     cell_max_capacity = 20*num_balls/grid_size**2 #20 times the mean
    #     cells = []
    #     for i in range(grid_size**2):
    #         j = 0
    #         cells.append([])
    #         while self.grid_array[int(i*cell_max_capacity + j)] != 0:
    #             cells[i].append(self.grid_array[i*cell_max_capacity + j])
    #             j += 1
    #     return cells
        print("-------------overlaps array-------------")
        print(self.ovelaps_array)

w = Work_in_GPU()




