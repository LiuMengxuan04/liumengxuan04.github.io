---
layout:     post
title:      "NS3学习笔记01 "
subtitle:   "tutorial third 样例分析"
date:       2025-10-19 19:10:00 +0800
author:     "Liu Mengxuan"
mathjax: true
header-img: "img/post-bg-miui6.jpg"
categories: [技术]
tags: [技术, ns3, Softmax, 网络仿真]
---

> 本文是 NS3 学习笔记的第一章节，examples/tutorial/third.cc 的样例分析

首先，我们概括一下这个示例的目标：**创建两个节点，用一条点对点链路将它们连接起来，在一个节点上运行一个 UDP 回显服务器，在另一个节点上运行一个 UDP 客户端，然后让客户端向服务器发送一个数据包，并接收返回的包。**

这就像用网线连接两台电脑，一台运行服务器软件，一台运行客户端软件来测试网络连通性。


## 前置概念
Node（节点）类：表示主机、路由器或者整体计算机的抽象；
Application（应用）类：表示网络应用程序；
Channel（通道）类：表示信道；
NetDevice（网络设备）类：表示Node上的网络通信设备及驱动程序，可以类比为网卡和网卡驱动的结合体。

### 节点

ns-3中基本计算设备被抽象为节点。我们可以将节点设想为一台可以添加各种功能的计算机。为了使一台计算机有效地工作，可以给它添加应用程序、协议栈、外设卡及驱动程序等。ns-3采用了与此相同的模型。由于ns-3是网络模拟器，而不是特定的Internet模拟器，因此故意不使用术语host，因为它与Internet及其协议密切相关。

### 应用

ns-3应用程序在ns-3节点上运行，应用程序是运行在node内部的**用户软件**，请注意，交换机的路由规则不属于用户软件，所以交换机实际上不需要部署应用。
ns-3期望开发人员在面向对象的编程意义上专门化Application类来创建新的应用程序（可以设计自己的Apllication类以特定的方式处理和分发数据包）。在ns-3中提供了名为UdpEchoClientApplication和UdpEchoServerApplication的Application类。这些应用程序组成了一个客户端/服务器应用程序集，用于生成和回显模拟的网络数据包。

### 通道

通常把网络中数据流流过的媒介称为信道。当你把以太网线插入到墙壁上的插孔时，你正通过信道将计算机与以太网连接。在ns-3的模拟环境中，你可以把节点连接到代表数据交换信道的对象上。在这里，基本的通信子网这一抽象概念被称为信道，在C++中用channel类来描述。

在本项目中，我们需要了解的主要是有线信道，NS3的有线信道一般分为CSMA（载波监听）和Pointopoint（点对点；p2p）两种。在数据中心的网络环境中，CSMA并不常见，我们主要还是考虑P2P的情况。

### 网络设备

计算机想要接入以太网，需要专门的外围设备和对应的驱动程序，也就是网卡和网卡驱动。在ns-3中，网络设备抽象为涵盖了软件驱动程序和模拟硬件的整体。网络设备被“安装”在节点中，以使节点能够通过信道与模拟中的其他节点通信。就像在真实计算机中一样，节点可以通过多个NetDevices连接到多个信道。
网络设备抽象在C++中由NetDevice类表示。 NetDevice类提供了管理Node和Channel对象连接的方法。net device 被安装在node中，使当前node能够和其他node构建网络（借助channel）。

## third.cc 示例讲解

`third.cc` 构建了一个混合网络拓扑，如下图所示：

```text
  //   Wifi 10.1.3.0
  //                 AP (接入点)
  //  *    *    *    *
  //  |    |    |    |    10.1.1.0 (点对点网络)
  // n5   n6   n7   n0 -------------- n1   n2   n3   n4
  //                   point-to-point  |    |    |    |
  //                                   ================
  //                                     LAN 10.1.2.0 (局域网)
```

*   它创建了一个**核心的点对点 (Point-to-Point) 链接**，连接了两个核心节点 `n0` 和 `n1`。这可以想象成连接两个核心路由器的光纤。
*   在节点 `n1` 上，它扩展出了一个 **CSMA 局域网** (可以理解为用交换机连接的以太网)，`n1` 和 `n2`, `n3`, `n4` 都在这个局域网里。
*   在节点 `n0` 上，它创建了一个 **WiFi 网络**，`n0` 作为 AP (无线接入点)，`n5`, `n6`, `n7` 作为普通的无线设备 (STA) 连接到这个 AP。
*   最终，它会在 WiFi 网络中的一个节点 (例如 `n7`) 上安装一个客户端**应用**，在 CSMA 局域网中的一个节点 (例如 `n4`) 上安装一个服务器**应用**。然后客户端会向服务器发送数据包，并验证整个网络的连通性。

---

## 概念与代码的对应关系

现在，我们来逐行分析代码，看看 `Node`、`Channel`、`NetDevice` 和 `Application` 是如何体现在代码中的。

### 1. Node (节点) - “创建计算机”

节点是所有网络功能的基础，是计算机的抽象。

*   **对应代码**:
    ```cpp
    62|    NodeContainer p2pNodes;
    63|    p2pNodes.Create(2); // 创建了 n0 和 n1
    ```
    这里创建了两个节点，用于构建点对点主干网络。`p2pNodes` 是一个容器，里面装着 `n0` 和 `n1` 这两台“裸机”。

    ```cpp
    72|    NodeContainer csmaNodes;
    73|    csmaNodes.Add(p2pNodes.Get(1)); // 把 n1 加入到新的容器中
    74|    csmaNodes.Create(nCsma); // 创建 n2, n3, n4 ...
    ```
    这里为 CSMA 局域网创建节点。注意，它首先将已经存在的 `n1` 节点加了进来，然后再创建了 `nCsma` 个新节点。这样，`n1` 就同时属于点对点网络和 CSMA 网络，成为了一个网关/路由器。

    ```cpp
    83|    NodeContainer wifiStaNodes;
    84|    wifiStaNodes.Create(nWifi); // 创建 n5, n6, n7 ...
    85|    NodeContainer wifiApNode = p2pNodes.Get(0); // 把 n0 指定为 AP 节点
    ```
    这里创建了 WiFi 网络的客户端节点。并且，它把已经存在的 `n0` 节点指定为 AP 节点，让 `n0` 也成为了一个连接点对点网络和 WiFi 网络的网关。

### 2. Channel (通道) & NetDevice (网络设备) - “安装网卡、连接网线”

这两者通常是紧密相关的。`NetDevice` 是安装在 `Node` 上的网卡，而 `Channel` 是连接这些网卡的“网线”或“无线信道”。

*   **对应代码 (点对点网络)**:
    ```cpp
    65|    PointToPointHelper pointToPoint; // 这是一个“助手”，用来简化配置
    66|    pointToPoint.SetDeviceAttribute("DataRate", StringValue("5Mbps")); // 设置网卡属性：速率5Mbps
    67|    pointToPoint.SetChannelAttribute("Delay", StringValue("2ms"));   // 设置通道属性：延迟2ms
    ...
    70|    p2pDevices = pointToPoint.Install(p2pNodes); // 核心步骤！
    ```
    `pointToPoint.Install(p2pNodes)` 这行代码做了两件大事：
    1.  **创建 NetDevice**: 在 `n0` 和 `n1` 上各“安装”了一块点对点网络设备（网卡）。
    2.  **创建 Channel**: 创建了一个点对点信道（网线），并将上面两块新创建的网卡连接到这个信道上。

*   **对应代码 (CSMA 局域网)**:
    ```cpp
    76|    CsmaHelper csma;
    77|    csma.SetChannelAttribute("DataRate", StringValue("100Mbps")); // 设置局域网总线速率
    ...
    81|    csmaDevices = csma.Install(csmaNodes);
    ```
    `csma.Install(csmaNodes)` 这行代码同样是创建了 `NetDevice` 和 `Channel`。它在 `csmaNodes` 容器里的所有节点（`n1`, `n2`, `n3`, `n4`）上都安装了一块 CSMA 网卡，然后创建了一个 CSMA 信道（可以想象成一个虚拟的集线器或交换机），并将所有这些网卡都连接到这个信道上。

*   **对应代码 (WiFi 网络)**:
    ```cpp
    87|    YansWifiChannelHelper channel = YansWifiChannelHelper::Default(); // Wifi 信道助手
    88|    YansWifiPhyHelper phy; // Wifi 物理层助手
    89|    phy.SetChannel(channel.Create()); // 创建一个无线信道
    ...
    98|    staDevices = wifi.Install(phy, mac, wifiStaNodes); // 在客户端节点上安装 Wifi 网卡
    ...
    102|   apDevices = wifi.Install(phy, mac, wifiApNode);    // 在 AP 节点上安装 AP 网卡
    ```
    WiFi 的创建过程更复杂，但本质是一样的。`channel.Create()` 创建了一个 `Channel`（无线环境）。`wifi.Install(...)` 则为所有 WiFi 节点创建并安装了 `NetDevice`（无线网卡），并让它们共享同一个无线信道。

### 3. 为之前创建好的网络基础设施（节点、设备、信道）进行配置，让它们能够真正地进行网络通信。

可以把它想象成 **“给电脑装系统、插网线上网、配IP地址、规划物理位置”** 的过程。

我们把它分解成三个主要步骤：

#### 1. 配置物理位置和移动模型 (104-126行)

这部分代码处理的是节点的**物理空间**属性，这对于无线网络模拟尤其重要，因为它决定了节点间的距离和信号强度。

*   **代码**:
    ```cpp
    104| MobilityHelper mobility;
    106| mobility.SetPositionAllocator("ns3::GridPositionAllocator", ...);
    ```
    **解释**: 这里首先定义了所有节点被创建时的初始位置。`GridPositionAllocator` 会像摆棋子一样，把节点一个个放置在一个网格里，避免它们在模拟开始时就重叠在一起。

*   **代码**:
    ```cpp
    120| mobility.SetMobilityModel("ns3::RandomWalk2dMobilityModel", ...);
    123| mobility.Install(wifiStaNodes);
    ```
    **解释**: 这里为所有的 WiFi 客户端 (`wifiStaNodes`) 设置了一个**移动模型**。`RandomWalk2dMobilityModel` 会让这些节点在模拟过程中像醉汉走路一样，在一个矩形区域内随机移动。`Install` 命令就是将这个“会随机移动”的属性赋予这些节点。

*   **代码**:
    ```cpp
    125| mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    126| mobility.Install(wifiApNode);
    ```
    **解释**: 这里为 WiFi 的 AP 节点 (`wifiApNode`) 设置了另一个移动模型。`ConstantPositionMobilityModel` 顾名思义，就是让这个节点**保持在原地不动**。这很符合现实，因为无线路由器通常是固定位置的。

**小结**: 这部分代码决定了节点在模拟世界中的“在哪里”以及“如何移动”。

---

#### 2. 安装协议栈 (128-131行)

仅仅有节点和网络设备（硬件）是不够的，计算机需要操作系统和网络协议（如TCP/IP）才能上网。这部分就是在做这个工作。

*   **代码**:
    ```cpp

    128|    InternetStackHelper stack;
    129|    stack.Install(csmaNodes);
    130|    stack.Install(wifiApNode);
    131|    stack.Install(wifiStaNodes);
    ```
    **解释**: `InternetStackHelper` 是一个非常重要的助手，它负责在节点上安装完整的互联网协议栈，包括 TCP, UDP, IP, ARP 等。
    `stack.Install(...)` 这个命令就相当于为 csmaNodes（`n1`, `n2`, `n3`, `n4`）、`wifiApNode`（`n0`）以及 `wifiStaNodes`（`n5`, `n6`, `n7`）这些“裸机”**安装了包含网络功能的操作系统**。执行完这一步，这些节点才具备了收发和处理IP数据包的能力。

**小结**: 这部分是赋予节点“联网智能”的关键。

---

#### 3. 分配IP地址 (133-145行)

安装完协议栈后，每个网络设备（网卡）都需要一个唯一的IP地址才能在网络中被识别和寻址。

*   **代码**:
    ```cpp
    133| Ipv4AddressHelper address;
    135| address.SetBase("10.1.1.0", "255.255.255.0");
    137| p2pInterfaces = address.Assign(p2pDevices);
    ```
    **解释**: 这里开始为点对点网络 (`p2pDevices`) 分配IP地址。`SetBase` 指定了网段为 `10.1.1.0`，子网掩码为 `255.255.255.0`。`address.Assign` 会自动地为 `p2pDevices` 容器中的设备（即 `n0` 和 `n1` 的点对点网卡）分配IP，通常第一个是 `10.1.1.1`，第二个是 `10.1.1.2`。

*   **代码**:
    ```cpp
    139| address.SetBase("10.1.2.0", "255.255.255.0");
    141| csmaInterfaces = address.Assign(csmaDevices);
    ```
    **解释**: 同样地，这里为 CSMA 局域网的设备分配 `10.1.2.0` 网段的IP地址。`n1` 的 CSMA 网卡会被分配 `10.1.2.1`，`n2` 的是 `10.1.2.2`，以此类推。

*   **代码**:
    ```cpp
    143| address.SetBase("10.1.3.0", "255.255.255.0");
    144| address.Assign(staDevices);
    145| address.Assign(apDevices);
    ```
    **解释**: 最后，为 WiFi 网络的所有设备（AP 和客户端）分配 `10.1.3.0` 网段的IP地址。

这部分完成了网络配置的最后一步，确保了每个节点上的每个网络接口都有了唯一的身份标识。

### 4. Application (应用) - “在计算机上运行程序”

当网络基础设施都搭建好之后（节点有了，网卡装了，网线也连了，IP地址也配好了），我们就需要在节点上运行程序来产生网络流量。

*   **对应代码**:
    ```cpp
    147|   UdpEchoServerHelper echoServer(9); // 创建一个服务器助手，监听9号端口
    ...
    149|   ApplicationContainer serverApps = echoServer.Install(csmaNodes.Get(nCsma)); // 在CSMA网络最后一个节点上安装服务器应用
    150|   serverApps.Start(Seconds(1.0)); // 在模拟时间的第1秒启动服务器
    151|   serverApps.Stop(Seconds(10.0)); // 在第10秒关闭
    ```
    这里，我们在 CSMA 网络的一个节点上安装并运行了 `UdpEchoServerApplication`。

    ```cpp
    153|   UdpEchoClientHelper echoClient(csmaInterfaces.GetAddress(nCsma), 9); // 创建客户端助手，并告诉它服务器的IP和端口
    ...
    158|   ApplicationContainer clientApps = echoClient.Install(wifiStaNodes.Get(nWifi - 1)); // 在WiFi网络最后一个节点上安装客户端应用
    159|   clientApps.Start(Seconds(2.0)); // 在第2秒启动客户端（比服务器晚一点）
    160|   clientApps.Stop(Seconds(10.0));
    ```
    相应地，在 WiFi 网络的一个节点上安装并运行了 `UdpEchoClientApplication`。这个客户端被配置为向服务器发送数据包。

## 总结

整个 `third.cc` 脚本的流程就像现实世界中组装和测试一个小型网络一样：
1.  **买电脑** (`NodeContainer::Create`) -> `Node`
2.  **买网卡和网线/路由器** (`XxxHelper::Install`) -> `NetDevice` & `Channel`
3.  **给电脑装系统、配IP** (`InternetStackHelper`, `Ipv4AddressHelper`)
4.  **在电脑上装软件（如FTP客户端和服务器）** (`UdpEchoXxxHelper::Install`) -> `Application`
5.  **开机运行、测试** (`Simulator::Run`)
