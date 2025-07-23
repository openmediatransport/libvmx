/*
* MIT License
* 
* Copyright (c) 2025 Open Media Transport Contributors
* 
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
* 
* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.
* 
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
* 
*/

#pragma once
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>

struct ThreadTask
{
	std::thread thread;
	std::queue<std::function<void()>> queue;
	std::mutex mtx;
	std::condition_variable cv;
	std::condition_variable complete;
	bool running = false;

	void Join()
	{
		std::unique_lock<std::mutex> lock(mtx);
		if (queue.size() > 0)
		{
			complete.wait(lock);
		}
	}

	void TaskLoop()
	{
		while (running)
		{			
			std::unique_lock<std::mutex> lock(mtx);
			if (queue.size() == 0)
			{
				complete.notify_all();
				cv.wait(lock);
			}
			if (!running) break;
			std::function<void()> func = NULL;
			if (queue.size() > 0)
			{
				func = queue.front();
				queue.pop();
			}
			if (func)
			{
				func();
			}

		}
	}
	void Initialize()
	{
		running = true;
		queue = std::queue<std::function<void()>>();
		thread = std::thread(&ThreadTask::TaskLoop, this);
	}
	void Push(std::function<void()> task)
	{
		{
			std::lock_guard<std::mutex> lock(mtx);
			queue.push(task);
		}
		cv.notify_all();
	}
	void Destroy()
	{
		{
			std::lock_guard<std::mutex> lock(mtx);
			running = false;
		}
		cv.notify_all();
		thread.join();
	}
};

struct ThreadTasks
{
	int numThreads;
	ThreadTask** tasks;
};

static ThreadTasks* CreateTasks(int numThreads)
{
	ThreadTasks* th = new ThreadTasks();
	th->numThreads = numThreads;
	th->tasks = new ThreadTask*[numThreads];
	for (int i = 0; i < numThreads; i++)
	{
		ThreadTask* task = new ThreadTask();
		th->tasks[i] = task;
		task->Initialize();
	}
	return th;
}

static void DestroyTasks(ThreadTasks* tasks)
{
	if (tasks)
	{
		for (int i = 0; i < tasks->numThreads; i++)
		{
			tasks->tasks[i]->Destroy();
			ThreadTask* task = tasks->tasks[i];
			delete task;
		}
		delete tasks;
	}
}

