#include <thread>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>

class threads
{
    std::vector<std::thread> thread_workers;
    std::queue<std::function<void()>> task;

    std::mutex mut;
    std::condition_variable cond;
    std::condition_variable finished_cond;

    bool stop_thread;
    size_t active_jobs = 0;

public:
    // This constructor creates a pool of worker threads that can execute small tensor tasks
    // concurrently while the main thread coordinates submission and completion.
    threads(size_t threads = 0) : stop_thread(false)
    {
        size_t num_threads = std::thread::hardware_concurrency();

        for (size_t ind = 0; ind < num_threads; ++ind)
        {
            thread_workers.emplace_back(std::thread([&]()
                                                    {
                while (true)
                {
                    std::unique_lock<std::mutex> lock(mut);

                    cond.wait(lock, [&] {
                        return stop_thread || !task.empty();
                    });

                    if (stop_thread && task.empty())
                        return;

                    auto tens_job = std::move(task.front());
                    task.pop();

                    lock.unlock();

                
                    tens_job();

                    // 🔥 FIX: protect active_jobs
                    {
                        std::unique_lock<std::mutex> lock(mut);
                        --active_jobs;

                        // 🔥 FIX: only check active_jobs
                        if (active_jobs == 0)
                            finished_cond.notify_all();
                    }
                } }));
        }
    }

    // This method submits a unit of work to the worker pool so the tensor kernels can run in
    // parallel when the workload is large enough to benefit from concurrency.
    template <typename FN>
    void enqueue(FN &&job)
    {
        std::unique_lock<std::mutex> lock(mut);

        task.push(std::forward<FN>(job));
        ++active_jobs;

        lock.unlock();

        cond.notify_all();
    }

    // This method blocks until every queued job has finished so callers can safely read the
    // results produced by the background workers.
    void wait()
    {
        std::unique_lock<std::mutex> lock(mut);

        finished_cond.wait(lock, [&]
                           { return active_jobs == 0; });
    }

    // This accessor reports how many worker threads are active in the pool.
    size_t num_threads() { return thread_workers.size(); }

    // This destructor shuts the pool down cleanly by stopping the workers and joining them
    // back to the calling thread once all pending tasks have been drained.
    ~threads()
    {
        std::unique_lock<std::mutex> lock(mut);
        stop_thread = true;
        lock.unlock();

        cond.notify_all();

        for (auto &worker : thread_workers)
            worker.join();
    }
};